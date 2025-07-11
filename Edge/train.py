import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import copy
from dataclasses import dataclass
from typing import Dict, Any, Optional

from models.fusion_net import FusionNet
from data.custom_dataset import get_dataloader, preprocess_batch
from utils.loss import get_loss_function
from utils.metrics import SegmentationMetrics
from config import Config
from validate import LightweightModelValidator


class LightweightTrainer:
    def __init__(self, config, experiment_name=None):
        self.config = config
        self.experiment_name = experiment_name or config.EXPERIMENT_NAME
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_dir = os.path.join(config.MODEL_SAVE_DIR, self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = FusionNet(config).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = get_loss_function(config)
        self.scaler = GradScaler() if config.MIXED_PRECISION else None

        self.train_loader = get_dataloader(config, 'train')
        self.val_loader = get_dataloader(config, 'val')

        self.best_miou = 0.0
        self.start_epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': []
        }

        print(f"Lightweight trainer initialized successfully")
        print(f"  Experiment name: {self.experiment_name}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {config.MIXED_PRECISION}")

    def _create_optimizer(self):
        if self.config.OPTIMIZER == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )

    def _create_scheduler(self):
        if self.config.SCHEDULER == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif self.config.SCHEDULER == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.NUM_EPOCHS // 3,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config.EARLY_STOPPING_PATIENCE // 2,
                factor=0.5
            )

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS}')

        for batch_idx, batch in enumerate(progress_bar):
            batch = preprocess_batch(batch, self.device)

            images = batch['image']
            labels = batch['label']
            edge_weights = batch.get('edge_weight', None)

            if self.scaler is not None:
                with autocast():
                    if hasattr(self.model, 'diw_module') and self.model.diw_module is not None:
                        logits, features = self.model(images, return_features=True)
                        high_freq_feat = features.get('high_freq_feat', None)
                        semantic_feat = features.get('semantic_feat', None)
                    else:
                        logits = self.model(images)
                        high_freq_feat = semantic_feat = None

                    loss = self.criterion(
                        logits, labels,
                        high_freq_feat=high_freq_feat,
                        semantic_feat=semantic_feat,
                        edge_weights=edge_weights
                    )

                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_NORM
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                if hasattr(self.model, 'diw_module') and self.model.diw_module is not None:
                    logits, features = self.model(images, return_features=True)
                    high_freq_feat = features.get('high_freq_feat', None)
                    semantic_feat = features.get('semantic_feat', None)
                else:
                    logits = self.model(images)
                    high_freq_feat = semantic_feat = None

                loss = self.criterion(
                    logits, labels,
                    high_freq_feat=high_freq_feat,
                    semantic_feat=semantic_feat,
                    edge_weights=edge_weights
                )

                loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_NORM
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

            progress_bar.set_postfix({
                'Loss': f'{loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            if (batch_idx + 1) % self.config.EMPTY_CACHE_INTERVAL == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        metrics = SegmentationMetrics(num_classes=self.config.NUM_CLASSES)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = preprocess_batch(batch, self.device)

                images = batch['image']
                labels = batch['label']
                edge_weights = batch.get('edge_weight', None)

                if hasattr(self.model, 'diw_module') and self.model.diw_module is not None:
                    logits, features = self.model(images, return_features=True)
                    high_freq_feat = features.get('high_freq_feat', None)
                    semantic_feat = features.get('semantic_feat', None)
                else:
                    logits = self.model(images)
                    high_freq_feat = semantic_feat = None

                loss = self.criterion(
                    logits, labels,
                    high_freq_feat=high_freq_feat,
                    semantic_feat=semantic_feat,
                    edge_weights=edge_weights
                )

                total_loss += loss.item()
                num_batches += 1

                preds = torch.argmax(logits, dim=1)
                metrics.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

        results = metrics.get_results()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return avg_loss, results

    def save_checkpoint(self, epoch, miou, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config,
            'training_history': self.training_history
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

            metrics_path = os.path.join(self.save_dir, 'best_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'miou': miou,
                    'training_time_minutes': (time.time() - self.train_start_time) / 60
                }, f, indent=4)

            print(f"Saved best model: mIoU = {miou:.4f}")

    def train(self):
        print(f"Starting training experiment: {self.experiment_name}")

        self.train_start_time = time.time()
        patience_counter = 0

        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)

            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_loss, val_results = self.validate(epoch)
                val_miou = val_results['Mean IoU']

                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_miou'].append(val_miou)

                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_miou)
                else:
                    self.scheduler.step()

                is_best = val_miou > self.best_miou
                if is_best:
                    self.best_miou = val_miou
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.save_checkpoint(epoch, val_miou, is_best)

                print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val mIoU: {val_miou:.4f}")
                print(f"  Best mIoU: {self.best_miou:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered: {self.config.EARLY_STOPPING_PATIENCE} epochs without improvement")
                    break

        training_time = time.time() - self.train_start_time
        print(f"\nTraining completed!")
        print(f"  Best mIoU: {self.best_miou:.4f}")
        print(f"  Training time: {training_time / 3600:.2f} hours")

        return self.best_miou


@dataclass
class OptimalConfig:
    name: str
    description: str
    scenario: str
    expected_params: str
    expected_miou: str
    expected_fps: str
    config_dict: Dict[str, Any]


class OptimalConfigManager:
    def __init__(self):
        self.configs = self._initialize_optimal_configs()

    def _initialize_optimal_configs(self) -> Dict[str, OptimalConfig]:
        configs = {}

        configs['ultra'] = OptimalConfig(
            name="Ours-Ultra",
            description="Ultra lightweight configuration for mobile and embedded devices",
            scenario="Mobile/Embedded Device",
            expected_params="~27K",
            expected_miou=">96.5%",
            expected_fps=">110",
            config_dict={
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': False,
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_RESIDUAL_CONNECTIONS': False,
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': 1.5,
                'USE_GRADIENT_LOSS': False,
                'LEARNING_RATE': 1e-3,
                'BATCH_SIZE': 16,
                'NUM_EPOCHS': 200,
                'EARLY_STOPPING_PATIENCE': 20,
                'EDGE_DETECTION_LAYERS': 1,
                'EDGE_ENHANCEMENT_STRENGTH': 2.5,
            }
        )

        configs['efficient'] = OptimalConfig(
            name="Ours-Efficient",
            description="Balanced efficiency and accuracy configuration for real-time applications",
            scenario="Real-time Applications",
            expected_params="~71K",
            expected_miou=">97.2%",
            expected_fps=">130",
            config_dict={
                'HIGH_FREQ_CHANNELS': 20,
                'SEMANTIC_CHANNELS': 24,
                'FUSION_CHANNELS': 32,
                'USE_DIW_MODULE': False,
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_RESIDUAL_CONNECTIONS': False,
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': 1.0,
                'USE_GRADIENT_LOSS': False,
                'LEARNING_RATE': 1e-3,
                'BATCH_SIZE': 12,
                'NUM_EPOCHS': 180,
                'EARLY_STOPPING_PATIENCE': 18,
                'EDGE_DETECTION_LAYERS': 2,
                'EDGE_ENHANCEMENT_STRENGTH': 2.0,
            }
        )

        configs['accurate'] = OptimalConfig(
            name="Ours-Accurate",
            description="High accuracy configuration for precision-demanding applications",
            scenario="High-precision Applications",
            expected_params="~99K",
            expected_miou=">97.4%",
            expected_fps=">100",
            config_dict={
                'HIGH_FREQ_CHANNELS': 24,
                'SEMANTIC_CHANNELS': 32,
                'FUSION_CHANNELS': 48,
                'USE_DIW_MODULE': True,
                'USE_SIMPLIFIED_DIW': True,
                'USE_PROGRESSIVE_REFINEMENT': True,
                'USE_RESIDUAL_CONNECTIONS': False,
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': 1.2,
                'USE_GRADIENT_LOSS': True,
                'GRADIENT_LOSS_WEIGHT': 0.5,
                'LEARNING_RATE': 8e-4,
                'BATCH_SIZE': 8,
                'NUM_EPOCHS': 200,
                'EARLY_STOPPING_PATIENCE': 25,
                'EDGE_DETECTION_LAYERS': 2,
                'EDGE_ENHANCEMENT_STRENGTH': 2.0,
            }
        )

        return configs

    def get_config(self, config_name: str) -> OptimalConfig:
        if config_name not in self.configs:
            available = list(self.configs.keys())
            raise ValueError(f"Configuration '{config_name}' does not exist. Available configs: {available}")

        return self.configs[config_name]

    def list_configs(self) -> Dict[str, str]:
        return {name: config.description for name, config in self.configs.items()}

    def create_training_config(self, config_name: str, base_config: Optional[Config] = None) -> Config:
        if base_config is None:
            base_config = Config()

        optimal_config = self.get_config(config_name)
        training_config = copy.deepcopy(base_config)

        for key, value in optimal_config.config_dict.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
            else:
                print(f"Warning: Configuration item {key} does not exist, skipping")

        training_config.EXPERIMENT_NAME = f"final_{optimal_config.name.lower().replace('-', '_')}"
        self._validate_training_config(training_config, optimal_config)

        return training_config

    def _validate_training_config(self, config: Config, optimal_config: OptimalConfig):
        errors = []
        warnings = []

        if config.SEMANTIC_CHANNELS % 2 != 0:
            errors.append(f"SEMANTIC_CHANNELS ({config.SEMANTIC_CHANNELS}) must be even")

        if config.FUSION_CHANNELS < max(config.HIGH_FREQ_CHANNELS, config.SEMANTIC_CHANNELS):
            errors.append(f"FUSION_CHANNELS should be >= max(HIGH_FREQ_CHANNELS, SEMANTIC_CHANNELS)")

        if config.USE_DIW_MODULE and config.HIGH_FREQ_CHANNELS <= 0:
            errors.append("HIGH_FREQ_CHANNELS must be > 0 when using DIW module")

        if config.BOUNDARY_LOSS_WEIGHT < 0:
            errors.append("BOUNDARY_LOSS_WEIGHT must be >= 0")

        if config.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be > 0")

        if config.NUM_EPOCHS <= 0:
            errors.append("NUM_EPOCHS must be > 0")

        if errors:
            print(f"Configuration {optimal_config.name} validation failed:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Configuration validation failed")

        if warnings:
            print(f"Configuration {optimal_config.name} warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        print(f"Configuration {optimal_config.name} validation passed")


class FinalModelTrainer:
    def __init__(self, save_dir: str = "./final_models"):
        self.save_dir = save_dir
        self.config_manager = OptimalConfigManager()
        self.results = {}

        os.makedirs(save_dir, exist_ok=True)

        print("Final model trainer initialized")
        print(f"Results save directory: {save_dir}")

    def train_single_config(self, config_name: str, base_config: Optional[Config] = None) -> Dict[str, Any]:
        optimal_config = self.config_manager.get_config(config_name)

        print(f"\n{'=' * 80}")
        print(f"Starting training final model: {optimal_config.name}")
        print(f"Configuration description: {optimal_config.description}")
        print(f"Target scenario: {optimal_config.scenario}")
        print(f"Expected performance: {optimal_config.expected_params} parameters, "
              f"{optimal_config.expected_miou} mIoU, {optimal_config.expected_fps} FPS")
        print(f"{'=' * 80}")

        training_config = self.config_manager.create_training_config(config_name, base_config)

        print(f"\nKey configurations:")
        key_configs = [
            'HIGH_FREQ_CHANNELS', 'SEMANTIC_CHANNELS', 'FUSION_CHANNELS',
            'USE_DIW_MODULE', 'USE_BOUNDARY_LOSS', 'BOUNDARY_LOSS_WEIGHT',
            'BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE'
        ]
        for key in key_configs:
            if hasattr(training_config, key):
                print(f"  {key}: {getattr(training_config, key)}")

        experiment_dir = os.path.join(training_config.MODEL_SAVE_DIR, training_config.EXPERIMENT_NAME)
        checkpoint_path = os.path.join(experiment_dir, "best_model.pth")

        if os.path.exists(checkpoint_path):
            print(f"\nFound existing training results: {checkpoint_path}")
            user_input = input("Do you want to retrain? (y/N): ").strip().lower()
            if user_input != 'y':
                print("Skipping training, loading existing results...")
                return self._load_existing_results(config_name, optimal_config, training_config, checkpoint_path)

        try:
            print(f"\nStarting training...")
            start_time = time.time()

            trainer = LightweightTrainer(training_config, training_config.EXPERIMENT_NAME)
            best_miou = trainer.train()

            training_time = time.time() - start_time

            print(f"\nTraining completed!")
            print(f"Best mIoU: {best_miou:.4f}")
            print(f"Training time: {training_time / 3600:.2f} hours")

            print(f"\nValidating performance...")
            validation_results = self._validate_final_model(training_config, checkpoint_path)

            final_results = {
                'config_name': config_name,
                'optimal_config': {
                    'name': optimal_config.name,
                    'description': optimal_config.description,
                    'scenario': optimal_config.scenario,
                    'expected_params': optimal_config.expected_params,
                    'expected_miou': optimal_config.expected_miou,
                    'expected_fps': optimal_config.expected_fps,
                },
                'training_results': {
                    'best_miou': best_miou,
                    'training_time_hours': training_time / 3600,
                    'training_time_minutes': training_time / 60,
                    'total_epochs': training_config.NUM_EPOCHS,
                },
                'performance_results': validation_results,
                'config_summary': self._get_config_summary(training_config),
                'checkpoint_path': checkpoint_path,
                'experiment_dir': experiment_dir,
            }

            self.results[config_name] = final_results
            self._save_individual_result(config_name, final_results)

            print(f"\n{optimal_config.name} training completed!")
            self._print_result_summary(final_results)

            return final_results

        except Exception as e:
            print(f"\nTraining failed: {e}")
            import traceback
            traceback.print_exc()

            error_result = {
                'config_name': config_name,
                'error': str(e),
                'config_summary': self._get_config_summary(training_config) if 'training_config' in locals() else {},
            }

            self.results[config_name] = error_result
            return error_result

        finally:
            torch.cuda.empty_cache()

    def _load_existing_results(self, config_name: str, optimal_config: OptimalConfig,
                               training_config: Config, checkpoint_path: str) -> Dict[str, Any]:
        try:
            experiment_dir = os.path.dirname(checkpoint_path)
            metrics_file = os.path.join(experiment_dir, "best_metrics.json")

            best_miou = 0.0
            training_time = 0.0

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                best_miou = saved_metrics.get('miou', 0.0)
                training_time = saved_metrics.get('training_time_minutes', 0.0)

            validation_results = self._validate_final_model(training_config, checkpoint_path)

            final_results = {
                'config_name': config_name,
                'optimal_config': {
                    'name': optimal_config.name,
                    'description': optimal_config.description,
                    'scenario': optimal_config.scenario,
                },
                'training_results': {
                    'best_miou': best_miou,
                    'training_time_minutes': training_time,
                    'loaded_from_cache': True,
                },
                'performance_results': validation_results,
                'config_summary': self._get_config_summary(training_config),
                'checkpoint_path': checkpoint_path,
                'experiment_dir': experiment_dir,
            }

            self.results[config_name] = final_results
            print(f"Loaded training results for {optimal_config.name}")

            return final_results

        except Exception as e:
            print(f"Failed to load existing results: {e}")
            raise e

    def _validate_final_model(self, config: Config, checkpoint_path: str) -> Dict[str, Any]:
        validator = LightweightModelValidator(config, checkpoint_path)

        model_info = validator.model_info
        speed_results = validator.benchmark_inference_speed(num_runs=100)
        memory_results = validator.measure_gpu_memory()
        comprehensive_results = validator.get_comprehensive_profile()

        return {
            'model_info': model_info,
            'speed_results': speed_results,
            'memory_results': memory_results,
            'comprehensive_results': comprehensive_results,
        }

    def _get_config_summary(self, config: Config) -> Dict[str, Any]:
        return {
            'HIGH_FREQ_CHANNELS': config.HIGH_FREQ_CHANNELS,
            'SEMANTIC_CHANNELS': config.SEMANTIC_CHANNELS,
            'FUSION_CHANNELS': config.FUSION_CHANNELS,
            'USE_DIW_MODULE': config.USE_DIW_MODULE,
            'USE_SIMPLIFIED_DIW': getattr(config, 'USE_SIMPLIFIED_DIW', False),
            'USE_BOUNDARY_LOSS': config.USE_BOUNDARY_LOSS,
            'BOUNDARY_LOSS_WEIGHT': getattr(config, 'BOUNDARY_LOSS_WEIGHT', 3.0),
            'USE_PROGRESSIVE_REFINEMENT': config.USE_PROGRESSIVE_REFINEMENT,
            'BATCH_SIZE': config.BATCH_SIZE,
            'NUM_EPOCHS': config.NUM_EPOCHS,
            'LEARNING_RATE': config.LEARNING_RATE,
        }

    def _save_individual_result(self, config_name: str, results: Dict[str, Any]):
        result_file = os.path.join(self.save_dir, f"{config_name}_final_results.json")

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)

        print(f"Results saved: {result_file}")

    def _print_result_summary(self, results: Dict[str, Any]):
        perf = results['performance_results']
        train = results['training_results']

        print(f"\n{results['optimal_config']['name']} final results:")
        print(f"  mIoU: {train['best_miou']:.4f}")
        print(f"  FPS: {perf['speed_results']['fps']:.1f}")
        print(f"  Parameters: {perf['model_info']['total_params']:,}")
        print(f"  Model size: {perf['model_info']['model_size_mb']:.2f} MB")
        print(f"  GPU memory: {perf['memory_results'].get('gpu_memory_peak_mb', 0):.1f} MB")
        print(f"  Inference time: {perf['speed_results']['avg_inference_time_ms']:.2f} ms")

        if 'comprehensive_results' in perf:
            comp = perf['comprehensive_results']
            print(f"  Overall efficiency: {comp.get('overall_efficiency', 0):.2f}")

    def train_all_configs(self, base_config: Optional[Config] = None) -> Dict[str, Any]:
        print(f"\nStarting training all optimal configurations")

        all_configs = ['ultra', 'efficient', 'accurate']

        for config_name in all_configs:
            print(f"\n" + "=" * 100)
            result = self.train_single_config(config_name, base_config)

            if 'error' in result:
                print(f"{config_name} training failed, continuing to next configuration")
            else:
                print(f"{config_name} training successful")

        self.generate_final_report()
        return self.results

    def generate_final_report(self):
        print(f"\nGenerating final training report...")

        final_results_file = os.path.join(self.save_dir, 'final_training_results.json')
        with open(final_results_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)

        self._generate_journal_table()
        self._generate_markdown_report()

        print(f"Final report saved to: {self.save_dir}")

    def _generate_journal_table(self):
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("No successful training results")
            return

        import pandas as pd

        table_data = []
        for config_name, result in successful_results.items():
            train_res = result['training_results']
            perf_res = result['performance_results']
            opt_config = result['optimal_config']

            row = {
                'Method': opt_config['name'],
                'Scenario': opt_config['scenario'],
                'mIoU': train_res['best_miou'],
                'Params(K)': perf_res['model_info']['total_params'] / 1000,
                'Size(MB)': perf_res['model_info']['model_size_mb'],
                'FPS': perf_res['speed_results']['fps'],
                'Inference(ms)': perf_res['speed_results']['avg_inference_time_ms'],
                'GPU_Memory(MB)': perf_res['memory_results'].get('gpu_memory_peak_mb', 0),
                'Training_Time(h)': train_res.get('training_time_hours', 0),
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)
        df = df.sort_values('Params(K)')

        csv_file = os.path.join(self.save_dir, 'final_models_comparison.csv')
        df.to_csv(csv_file, index=False)

        print(f"\nFinal models comparison table:")
        print(df.to_string(index=False, float_format='%.3f'))



def train_single_optimal_config(config_name: str, base_config: Optional[Config] = None,
                                save_dir: str = "./final_models") -> Dict[str, Any]:
    trainer = FinalModelTrainer(save_dir)
    return trainer.train_single_config(config_name, base_config)


def train_all_optimal_configs(base_config: Optional[Config] = None,
                              save_dir: str = "./final_models") -> Dict[str, Any]:
    trainer = FinalModelTrainer(save_dir)
    return trainer.train_all_configs(base_config)


def compare_with_expectations(results: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    manager = OptimalConfigManager()
    comparison = {}

    for config_name, result in results.items():
        if 'error' in result:
            continue

        optimal_config = manager.get_config(config_name)
        train_res = result['training_results']
        perf_res = result['performance_results']

        actual_params = perf_res['model_info']['total_params'] / 1000
        actual_miou = train_res['best_miou']
        actual_fps = perf_res['speed_results']['fps']

        expected_params_str = optimal_config.expected_params.replace('~', '').replace('K', '')
        expected_miou_str = optimal_config.expected_miou.replace('>', '').replace('%', '')
        expected_fps_str = optimal_config.expected_fps.replace('>', '')

        try:
            expected_params = float(expected_params_str)
            expected_miou = float(expected_miou_str) / 100
            expected_fps = float(expected_fps_str)

            comparison[config_name] = {
                'params': f"Actual: {actual_params:.1f}K, Expected: {expected_params}K, {'Pass' if actual_params <= expected_params * 1.2 else 'Fail'}",
                'miou': f"Actual: {actual_miou:.3f}, Expected: >{expected_miou:.3f}, {'Pass' if actual_miou >= expected_miou else 'Fail'}",
                'fps': f"Actual: {actual_fps:.1f}, Expected: >{expected_fps}, {'Pass' if actual_fps >= expected_fps else 'Fail'}"
            }
        except:
            comparison[config_name] = {'error': 'Failed to parse expected values'}

    return comparison


def generate_deployment_configs(results: Dict[str, Any], save_dir: str) -> Dict[str, str]:
    deployment_configs = {}

    for config_name, result in results.items():
        if 'error' in result:
            continue

        config_summary = result['config_summary']
        perf_info = result['performance_results']['model_info']

        deployment_config = {
            'model_name': result['optimal_config']['name'],
            'scenario': result['optimal_config']['scenario'],
            'checkpoint_path': result['checkpoint_path'],
            'model_params': perf_info['total_params'],
            'model_size_mb': perf_info['model_size_mb'],
            'expected_fps': result['performance_results']['speed_results']['fps'],
            'config': config_summary,
            'deployment_notes': {
                'memory_requirement': f"{result['performance_results']['memory_results'].get('gpu_memory_peak_mb', 0):.1f} MB GPU",
                'inference_time': f"{result['performance_results']['speed_results']['avg_inference_time_ms']:.2f} ms",
                'batch_processing': config_summary['BATCH_SIZE'],
            }
        }

        deploy_file = os.path.join(save_dir, f"{config_name}_deployment_config.json")
        with open(deploy_file, 'w') as f:
            json.dump(deployment_config, f, indent=4)

        deployment_configs[config_name] = deploy_file

    return deployment_configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Final model training - Three optimal configurations')
    parser.add_argument('--config', type=str,
                        choices=['ultra', 'efficient', 'accurate', 'all'],
                        default='all',
                        help='Configuration to train (ultra/efficient/accurate/all)')
    parser.add_argument('--save_dir', type=str, default='./final_models',
                        help='Results save directory')
    parser.add_argument('--base_config', type=str, default=None,
                        help='Base configuration file path')
    parser.add_argument('--compare_expectations', action='store_true',
                        help='Compare actual results with expectations')
    parser.add_argument('--generate_deployment', action='store_true',
                        help='Generate deployment configuration files')

    args = parser.parse_args()

    if args.base_config:
        with open(args.base_config, 'r') as f:
            base_config_dict = json.load(f)
        base_config = Config()
        for key, value in base_config_dict.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    else:
        base_config = Config()

    manager = OptimalConfigManager()
    print("Available optimal configurations:")
    for name, description in manager.list_configs().items():
        config_obj = manager.get_config(name)
        print(f"  {name}: {description}")
        print(
            f"    Expected performance: {config_obj.expected_params} parameters, {config_obj.expected_miou} mIoU, {config_obj.expected_fps} FPS")

    print(f"\nStarting training configuration: {args.config}")

    if args.config == 'all':
        results = train_all_optimal_configs(base_config, args.save_dir)
    else:
        result = train_single_optimal_config(args.config, base_config, args.save_dir)
        results = {args.config: result}

    if args.compare_expectations and results:
        print(f"\nActual vs Expected Results Comparison:")
        comparison = compare_with_expectations(results)
        for config_name, comp in comparison.items():
            print(f"\n{config_name}:")
            for metric, result in comp.items():
                print(f"  {metric}: {result}")

    if args.generate_deployment and results:
        print(f"\nGenerating deployment configuration files...")
        deployment_configs = generate_deployment_configs(results, args.save_dir)
        print("Deployment configuration files:")
        for config_name, file_path in deployment_configs.items():
            print(f"  {config_name}: {file_path}")

    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}

    print(f"\nTraining completion summary:")
    print(f"Successful: {len(successful_results)} configurations")
    print(f"Failed: {len(failed_results)} configurations")

    if successful_results:
        print(f"\nSuccessful configurations:")
        for name, result in successful_results.items():
            train_res = result['training_results']
            perf_res = result['performance_results']
            print(f"  {name}: mIoU={train_res['best_miou']:.4f}, "
                  f"params={perf_res['model_info']['total_params']:,}, "
                  f"FPS={perf_res['speed_results']['fps']:.1f}")

    if failed_results:
        print(f"\nFailed configurations:")
        for name, result in failed_results.items():
            print(f"  {name}: {result['error']}")

    print(f"\nAll results saved in: {args.save_dir}")
    print("Ready to start writing the journal paper!")
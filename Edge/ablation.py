import os
import json
import time
import copy
import torch
import pandas as pd
import numpy as np

from config import Config
from train import LightweightTrainer
from validate import LightweightModelValidator
from utils.metrics import ModelComparator


class LightweightAblationStudy:
    def __init__(self, base_config, save_dir="./ablation_results"):
        self.base_config = base_config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.results = {}

        print("Lightweight ablation study initialized - Phase 1 optimized version")
        print(f"Base configuration summary: {self._get_config_summary(base_config)}")
        print(f"Results save directory: {save_dir}")

    def _get_config_summary(self, config):
        return {
            'HIGH_FREQ_CHANNELS': config.HIGH_FREQ_CHANNELS,
            'SEMANTIC_CHANNELS': config.SEMANTIC_CHANNELS,
            'FUSION_CHANNELS': config.FUSION_CHANNELS,
            'USE_DIW_MODULE': config.USE_DIW_MODULE,
            'DIW_MODULE_TYPE': getattr(config, 'DIW_MODULE_TYPE', 'standard'),
            'USE_PROGRESSIVE_REFINEMENT': config.USE_PROGRESSIVE_REFINEMENT,
            'USE_BOUNDARY_LOSS': config.USE_BOUNDARY_LOSS,
            'LOSS_TYPE': getattr(config, 'LOSS_TYPE', 'combined'),
            'BATCH_SIZE': config.BATCH_SIZE,
        }

    def create_config_variant(self, variant_name, modifications):
        config = copy.deepcopy(self.base_config)

        for key, value in modifications.items():
            setattr(config, key, value)

        config.BATCH_SIZE = min(config.BATCH_SIZE, 16)
        config.GRADIENT_ACCUMULATION_STEPS = max(4, config.GRADIENT_ACCUMULATION_STEPS)
        config.MIXED_PRECISION = True
        config.EMPTY_CACHE_INTERVAL = 10
        config.NUM_WORKERS = 0
        config.PIN_MEMORY = False

        config.EXPERIMENT_NAME = f"ablation_{variant_name}"

        return config

    def run_single_experiment(self, variant_name, config, max_epochs=30):
        print(f"\n{'=' * 60}")
        print(f"Ablation experiment: {variant_name}")
        print(f"{'=' * 60}")

        model_checkpoint = os.path.join(
            config.MODEL_SAVE_DIR, f"ablation_{variant_name}", "best_model.pth"
        )

        if os.path.exists(model_checkpoint):
            print(f"Found existing model checkpoint, skipping training: {variant_name}")

            try:
                validator = LightweightModelValidator(config, model_checkpoint)

                metrics_file = os.path.join(
                    config.MODEL_SAVE_DIR, f"ablation_{variant_name}", "best_metrics.json"
                )

                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        saved_metrics = json.load(f)
                    best_miou = saved_metrics.get('miou', 0.0)
                    training_time = saved_metrics.get('training_time_minutes', 0.0)
                else:
                    best_miou = 0.0
                    training_time = 0.0

                speed_results = validator.benchmark_inference_speed(num_runs=50)
                memory_results = validator.measure_gpu_memory()
                model_info = validator.model_info

                self.results[variant_name] = {
                    'config_summary': self._get_config_summary(config),
                    'best_miou': best_miou,
                    'training_time_minutes': training_time,
                    'model_info': model_info,
                    'performance': {
                        'fps': speed_results['fps'],
                        'inference_time_ms': speed_results['avg_inference_time_ms'],
                        'gpu_memory_mb': memory_results.get('gpu_memory_peak_mb', 0)
                    },
                    'efficiency_metrics': self._calculate_efficiency_metrics(
                        best_miou, model_info, speed_results
                    ),
                    'skipped': True
                }

                print(f"Loaded experiment {variant_name} results:")
                print(f"  Best mIoU: {best_miou:.4f}")
                print(f"  FPS: {speed_results['fps']:.1f}")
                print(f"  Parameters: {model_info['total_params']:,}")

                return

            except Exception as e:
                print(f"Failed to load existing results, retraining: {e}")

        config_summary = self._get_config_summary(config)
        for key, value in config_summary.items():
            print(f"  {key}: {value}")

        torch.cuda.empty_cache()

        try:
            trainer = LightweightTrainer(config, f"ablation_{variant_name}")

            original_epochs = config.NUM_EPOCHS
            config.NUM_EPOCHS = max_epochs

            start_time = time.time()
            best_miou = trainer.train()
            training_time = time.time() - start_time

            config.NUM_EPOCHS = original_epochs

            if os.path.exists(model_checkpoint):
                validator = LightweightModelValidator(config, model_checkpoint)

                speed_results = validator.benchmark_inference_speed(num_runs=50)
                memory_results = validator.measure_gpu_memory()
                model_info = validator.model_info

                self.results[variant_name] = {
                    'config_summary': config_summary,
                    'best_miou': best_miou,
                    'training_time_minutes': training_time / 60,
                    'model_info': model_info,
                    'performance': {
                        'fps': speed_results['fps'],
                        'inference_time_ms': speed_results['avg_inference_time_ms'],
                        'gpu_memory_mb': memory_results.get('gpu_memory_peak_mb', 0)
                    },
                    'efficiency_metrics': self._calculate_efficiency_metrics(
                        best_miou, model_info, speed_results
                    )
                }

                print(f"Experiment {variant_name} completed:")
                print(f"  Best mIoU: {best_miou:.4f}")
                print(f"  Training time: {training_time / 60:.1f} minutes")
                print(f"  FPS: {speed_results['fps']:.1f}")
                print(f"  Parameters: {model_info['total_params']:,}")

            else:
                print(f"Model checkpoint not found: {model_checkpoint}")
                self.results[variant_name] = {
                    'error': 'Model checkpoint not found',
                    'config_summary': config_summary
                }

        except Exception as e:
            print(f"Experiment {variant_name} failed: {e}")
            import traceback
            traceback.print_exc()

            self.results[variant_name] = {
                'error': str(e),
                'config_summary': config_summary
            }

        finally:
            torch.cuda.empty_cache()

    def _calculate_efficiency_metrics(self, miou, model_info, speed_results):
        params_m = model_info['total_params'] / 1e6
        fps = speed_results['fps']
        memory_mb = speed_results.get('gpu_memory_mb', 1)

        return {
            'accuracy_per_param': miou / params_m if params_m > 0 else 0,
            'speed_per_param': fps / params_m if params_m > 0 else 0,
            'memory_efficiency': fps / memory_mb if memory_mb > 0 else 0,
            'overall_efficiency': (miou * fps) / params_m if params_m > 0 else 0
        }

    def run_ultra_lightweight_optimization(self):
        print(f"\n{'=' * 50}")
        print("Phase 1: Ultra lightweight optimization ablation study")
        print('=' * 50)

        ultra_variants = [
            ("ultra_light_v2", {
                'HIGH_FREQ_CHANNELS': self.base_config.ULTRA_LIGHT_V2_HIGH_FREQ_CHANNELS,
                'SEMANTIC_CHANNELS': self.base_config.ULTRA_LIGHT_V2_SEMANTIC_CHANNELS,
                'FUSION_CHANNELS': self.base_config.ULTRA_LIGHT_V2_FUSION_CHANNELS,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': True,
                'LOSS_TYPE': 'stable',
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_STABLE,
                'EDGE_DETECTION_LAYERS': self.base_config.EDGE_DETECTION_LAYERS_MINIMAL,
            }),
            ("ultra_light_v3", {
                'HIGH_FREQ_CHANNELS': self.base_config.ULTRA_LIGHT_V3_HIGH_FREQ_CHANNELS,
                'SEMANTIC_CHANNELS': self.base_config.ULTRA_LIGHT_V3_SEMANTIC_CHANNELS,
                'FUSION_CHANNELS': self.base_config.ULTRA_LIGHT_V3_FUSION_CHANNELS,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': True,
                'LOSS_TYPE': 'stable',
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_STABLE,
                'EDGE_DETECTION_LAYERS': self.base_config.EDGE_DETECTION_LAYERS_MINIMAL,
            }),
            ("minimal_extreme", {
                'HIGH_FREQ_CHANNELS': self.base_config.MINIMAL_HIGH_FREQ_CHANNELS,
                'SEMANTIC_CHANNELS': self.base_config.MINIMAL_SEMANTIC_CHANNELS,
                'FUSION_CHANNELS': self.base_config.MINIMAL_FUSION_CHANNELS,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'minimal',
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': True,
                'LOSS_TYPE': 'stable',
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_MINIMAL,
                'EDGE_DETECTION_LAYERS': self.base_config.EDGE_DETECTION_LAYERS_ULTRA_MINIMAL,
                'EDGE_ENHANCEMENT_TYPE': 'minimal',
                'SEG_HEAD_TYPE': 'minimal',
            }),
            ("ultra_light_no_diw", {
                'HIGH_FREQ_CHANNELS': self.base_config.ULTRA_LIGHT_V2_HIGH_FREQ_CHANNELS,
                'SEMANTIC_CHANNELS': self.base_config.ULTRA_LIGHT_V2_SEMANTIC_CHANNELS,
                'FUSION_CHANNELS': self.base_config.ULTRA_LIGHT_V2_FUSION_CHANNELS,
                'USE_DIW_MODULE': False,
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': False,
                'LOSS_TYPE': 'simplified',
                'EDGE_DETECTION_LAYERS': self.base_config.EDGE_DETECTION_LAYERS_MINIMAL,
            }),
        ]

        for variant_name, modifications in ultra_variants:
            config = self.create_config_variant(f"ultra_{variant_name}", modifications)
            self.run_single_experiment(f"ultra_{variant_name}", config)

    def run_loss_function_ablation(self):
        print(f"\n{'=' * 50}")
        print("Phase 1: Loss function stability ablation study")
        print('=' * 50)

        loss_variants = [
            ("stable_loss", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_STABLE,
                'LOSS_TYPE': 'stable',
                'USE_GRADIENT_LOSS_MINIMAL': True,
                'GRADIENT_LOSS_WEIGHT': self.base_config.GRADIENT_LOSS_WEIGHT_MINIMAL,
            }),
            ("minimal_loss", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_MINIMAL,
                'LOSS_TYPE': 'stable',
                'USE_GRADIENT_LOSS_MINIMAL': True,
                'GRADIENT_LOSS_WEIGHT': self.base_config.GRADIENT_LOSS_WEIGHT_ULTRA_MINIMAL,
            }),
            ("ultra_minimal_loss", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'minimal',
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_ULTRA_MINIMAL,
                'LOSS_TYPE': 'stable',
                'USE_GRADIENT_LOSS_MINIMAL': False,
            }),
            ("simplified_only", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'simplified',
                'USE_BOUNDARY_LOSS': False,
                'LOSS_TYPE': 'simplified',
            }),
        ]

        for variant_name, modifications in loss_variants:
            config = self.create_config_variant(f"loss_{variant_name}", modifications)
            self.run_single_experiment(f"loss_{variant_name}", config)

    def run_diw_module_comparison(self):
        print(f"\n{'=' * 50}")
        print("Phase 1: DIW module type comparison study")
        print('=' * 50)

        base_channels = {
            'HIGH_FREQ_CHANNELS': 16,
            'SEMANTIC_CHANNELS': 20,
            'FUSION_CHANNELS': 24,
            'USE_BOUNDARY_LOSS': True,
            'LOSS_TYPE': 'stable',
            'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_STABLE,
        }

        diw_variants = [
            ("diw_standard", {
                **base_channels,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'standard',
            }),
            ("diw_ultra_lightweight", {
                **base_channels,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
            }),
            ("diw_minimal", {
                **base_channels,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'minimal',
            }),
            ("diw_simplified", {
                **base_channels,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'simplified',
            }),
            ("no_diw_baseline", {
                **base_channels,
                'USE_DIW_MODULE': False,
            }),
        ]

        for variant_name, modifications in diw_variants:
            config = self.create_config_variant(f"diw_{variant_name}", modifications)
            self.run_single_experiment(f"diw_{variant_name}", config)

    def run_channel_sensitivity_analysis(self):
        print(f"\n{'=' * 50}")
        print("Phase 1: Channel sensitivity analysis")
        print('=' * 50)

        channel_combinations = [
            (8, 12, 16),
            (10, 14, 18),
            (12, 16, 20),
            (14, 18, 22),
            (16, 20, 24),
            (18, 22, 26),
            (20, 24, 32),
        ]

        for i, (hf_ch, sem_ch, fus_ch) in enumerate(channel_combinations):
            variant_name = f"channels_{hf_ch}_{sem_ch}_{fus_ch}"
            modifications = {
                'HIGH_FREQ_CHANNELS': hf_ch,
                'SEMANTIC_CHANNELS': sem_ch,
                'FUSION_CHANNELS': fus_ch,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_BOUNDARY_LOSS': True,
                'LOSS_TYPE': 'stable',
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_STABLE,
            }

            config = self.create_config_variant(variant_name, modifications)
            self.run_single_experiment(variant_name, config, max_epochs=25)

    def run_edge_enhancement_ablation(self):
        print(f"\n{'=' * 50}")
        print("Phase 1: Edge enhancement mechanism ablation study")
        print('=' * 50)

        edge_variants = [
            ("edge_none", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_BOUNDARY_LOSS': False,
                'LOSS_TYPE': 'simplified',
                'EDGE_DETECTION_LAYERS': 0,
                'EDGE_ENHANCEMENT_TYPE': 'minimal',
            }),
            ("edge_minimal", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_ULTRA_MINIMAL,
                'LOSS_TYPE': 'stable',
                'EDGE_DETECTION_LAYERS': 1,
                'EDGE_ENHANCEMENT_TYPE': 'minimal',
            }),
            ("edge_standard", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
                'USE_DIW_MODULE': True,
                'DIW_MODULE_TYPE': 'ultra_lightweight',
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': self.base_config.BOUNDARY_LOSS_WEIGHT_STABLE,
                'LOSS_TYPE': 'stable',
                'EDGE_DETECTION_LAYERS': 2,
                'EDGE_ENHANCEMENT_TYPE': 'standard',
            }),
        ]

        for variant_name, modifications in edge_variants:
            config = self.create_config_variant(f"edge_{variant_name}", modifications)
            self.run_single_experiment(f"edge_{variant_name}", config)

    def run_lightweight_channel_ablation(self):
        print(f"\n{'=' * 50}")
        print("Lightweight channel configuration ablation")
        print('=' * 50)

        channel_variants = [
            ("ultra_light", {
                'HIGH_FREQ_CHANNELS': 16,
                'SEMANTIC_CHANNELS': 20,
                'FUSION_CHANNELS': 24,
            }),
            ("very_light", {
                'HIGH_FREQ_CHANNELS': 20,
                'SEMANTIC_CHANNELS': 24,
                'FUSION_CHANNELS': 32,
            }),
            ("light", {
                'HIGH_FREQ_CHANNELS': 24,
                'SEMANTIC_CHANNELS': 32,
                'FUSION_CHANNELS': 48,
            }),
            ("standard", {
                'HIGH_FREQ_CHANNELS': 32,
                'SEMANTIC_CHANNELS': 40,
                'FUSION_CHANNELS': 64,
            }),
        ]

        for variant_name, modifications in channel_variants:
            config = self.create_config_variant(f"channel_{variant_name}", modifications)
            self.run_single_experiment(f"channel_{variant_name}", config)

    def run_lightweight_module_ablation(self):
        print(f"\n{'=' * 50}")
        print("Lightweight module ablation study")
        print('=' * 50)

        module_variants = [
            ("baseline", {
                'USE_DIW_MODULE': False,
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': False,
                'USE_GRADIENT_LOSS': False,
            }),
            ("diw_only", {
                'USE_DIW_MODULE': True,
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': False,
                'USE_GRADIENT_LOSS': False,
            }),
            ("diw_progressive", {
                'USE_DIW_MODULE': True,
                'USE_PROGRESSIVE_REFINEMENT': True,
                'USE_BOUNDARY_LOSS': False,
                'USE_GRADIENT_LOSS': False,
            }),
            ("diw_boundary", {
                'USE_DIW_MODULE': True,
                'USE_PROGRESSIVE_REFINEMENT': False,
                'USE_BOUNDARY_LOSS': True,
                'USE_GRADIENT_LOSS': False,
            }),
            ("full_model", {
                'USE_DIW_MODULE': True,
                'USE_PROGRESSIVE_REFINEMENT': True,
                'USE_BOUNDARY_LOSS': True,
                'USE_GRADIENT_LOSS': True,
            }),
        ]

        for variant_name, modifications in module_variants:
            config = self.create_config_variant(f"module_{variant_name}", modifications)
            self.run_single_experiment(f"module_{variant_name}", config)

    def run_edge_processing_ablation(self):
        print(f"\n{'=' * 50}")
        print("Edge processing ablation study")
        print('=' * 50)

        edge_variants = [
            ("no_edge", {
                'USE_BOUNDARY_LOSS': False,
                'BOUNDARY_LOSS_WEIGHT': 0,
                'EDGE_ENHANCEMENT_STRENGTH': 1.0,
                'EDGE_DETECTION_LAYERS': 1,
            }),
            ("basic_edge", {
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': 1.5,
                'EDGE_ENHANCEMENT_STRENGTH': 1.5,
                'EDGE_DETECTION_LAYERS': 2,
            }),
            ("enhanced_edge", {
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': 3.0,
                'EDGE_ENHANCEMENT_STRENGTH': 2.0,
                'EDGE_DETECTION_LAYERS': 2,
            }),
            ("strong_edge", {
                'USE_BOUNDARY_LOSS': True,
                'BOUNDARY_LOSS_WEIGHT': 5.0,
                'EDGE_ENHANCEMENT_STRENGTH': 3.0,
                'EDGE_DETECTION_LAYERS': 3,
            }),
        ]

        for variant_name, modifications in edge_variants:
            config = self.create_config_variant(f"edge_{variant_name}", modifications)
            self.run_single_experiment(f"edge_{variant_name}", config)

    def run_phase1_comprehensive_study(self, max_epochs=30):
        print(f"\n{'=' * 60}")
        print("Phase 1: Comprehensive lightweight optimization study")
        print('=' * 60)

        start_time = time.time()

        try:
            self.run_ultra_lightweight_optimization()
            self.run_loss_function_ablation()
            self.run_diw_module_comparison()
            self.run_channel_sensitivity_analysis()
            self.run_edge_enhancement_ablation()

            self.generate_simple_report()

            total_time = time.time() - start_time
            print(f"\nPhase 1 comprehensive optimization study completed, total time: {total_time / 3600:.1f} hours")

        except KeyboardInterrupt:
            print(f"\nPhase 1 study interrupted by user")
            if len(self.results) > 0:
                print("Generating partial results report...")
                self.generate_simple_report()
        except Exception as e:
            print(f"\nPhase 1 study failed: {e}")
            if len(self.results) > 0:
                print("Generating partial results report...")
                self.generate_simple_report()

    def generate_simple_report(self):
        print(f"\n{'=' * 50}")
        print("Generating Phase 1 optimization study report")
        print('=' * 50)

        with open(os.path.join(self.save_dir, 'phase1_results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)

        self._generate_comparison_table()

        print(f"Phase 1 study report saved to: {self.save_dir}")

    def _generate_comparison_table(self):
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("No successful experiment results for comparison")
            return

        table_data = []
        for variant_name, result in successful_results.items():
            row = {
                'Variant': variant_name,
                'mIoU': result.get('best_miou', 0),
                'Params(M)': result['model_info']['total_params'] / 1e6,
                'Size(MB)': result['model_info']['model_size_mb'],
                'FPS': result['performance']['fps'],
                'GPU_Mem(MB)': result['performance']['gpu_memory_mb'],
                'Train_Time(min)': result.get('training_time_minutes', 0),
                'Efficiency': result['efficiency_metrics']['overall_efficiency'],
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)
        df = df.sort_values('Efficiency', ascending=False)

        df.to_csv(os.path.join(self.save_dir, 'comparison_table.csv'), index=False)

        print(f"\nAblation study results TOP 10:")
        print(df.head(10).to_string(index=False, float_format='%.3f'))

    def generate_ablation_report(self):
        print(f"\n{'=' * 50}")
        print("Generating ablation study report")
        print('=' * 50)

        with open(os.path.join(self.save_dir, 'ablation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)

        self._generate_comparison_table()

        print(f"Ablation study report saved to: {self.save_dir}")

    def run_full_lightweight_study(self, max_epochs=50):
        print("Starting complete lightweight ablation study")
        print(f"Results will be saved to: {self.save_dir}")

        start_time = time.time()

        try:
            self.run_lightweight_channel_ablation()
            self.run_lightweight_module_ablation()
            self.run_edge_processing_ablation()

            self.run_ultra_lightweight_optimization()
            self.run_loss_function_ablation()
            self.run_diw_module_comparison()

            self.generate_ablation_report()

            total_time = time.time() - start_time
            print(f"\nLightweight ablation study completed, total time: {total_time / 3600:.1f} hours")

        except KeyboardInterrupt:
            print("\nAblation study interrupted by user")
            if len(self.results) > 0:
                print("Generating partial results report...")
                self.generate_ablation_report()
        except Exception as e:
            print(f"\nAblation study failed: {e}")
            if len(self.results) > 0:
                print("Generating partial results report...")
                self.generate_ablation_report()


def run_phase1_quick_ablation(config, save_dir=None):
    print("Starting Phase 1 quick lightweight ablation study")

    if save_dir is None:
        save_dir = "./phase1_results"

    ablation = LightweightAblationStudy(config, save_dir)

    phase1_variants = [
        ("baseline_optimized", {
            'HIGH_FREQ_CHANNELS': 24,
            'SEMANTIC_CHANNELS': 32,
            'FUSION_CHANNELS': 48,
            'USE_DIW_MODULE': False,
            'USE_PROGRESSIVE_REFINEMENT': False,
            'USE_BOUNDARY_LOSS': False,
            'LOSS_TYPE': 'simplified',
        }),
        ("ultra_light_v2", {
            'HIGH_FREQ_CHANNELS': config.ULTRA_LIGHT_V2_HIGH_FREQ_CHANNELS,
            'SEMANTIC_CHANNELS': config.ULTRA_LIGHT_V2_SEMANTIC_CHANNELS,
            'FUSION_CHANNELS': config.ULTRA_LIGHT_V2_FUSION_CHANNELS,
            'USE_DIW_MODULE': True,
            'DIW_MODULE_TYPE': 'ultra_lightweight',
            'USE_PROGRESSIVE_REFINEMENT': False,
            'USE_BOUNDARY_LOSS': True,
            'LOSS_TYPE': 'stable',
            'BOUNDARY_LOSS_WEIGHT': config.BOUNDARY_LOSS_WEIGHT_STABLE,
        }),
        ("minimal_extreme", {
            'HIGH_FREQ_CHANNELS': config.MINIMAL_HIGH_FREQ_CHANNELS,
            'SEMANTIC_CHANNELS': config.MINIMAL_SEMANTIC_CHANNELS,
            'FUSION_CHANNELS': config.MINIMAL_FUSION_CHANNELS,
            'USE_DIW_MODULE': True,
            'DIW_MODULE_TYPE': 'minimal',
            'USE_PROGRESSIVE_REFINEMENT': False,
            'USE_BOUNDARY_LOSS': True,
            'LOSS_TYPE': 'stable',
            'BOUNDARY_LOSS_WEIGHT': config.BOUNDARY_LOSS_WEIGHT_MINIMAL,
            'EDGE_ENHANCEMENT_TYPE': 'minimal',
            'SEG_HEAD_TYPE': 'minimal',
        }),
    ]

    for variant_name, modifications in phase1_variants:
        config_variant = ablation.create_config_variant(variant_name, modifications)
        ablation.run_single_experiment(variant_name, config_variant, max_epochs=25)

    ablation.generate_simple_report()

    print("Phase 1 quick ablation study completed!")
    return ablation.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1 lightweight network ablation study')
    parser.add_argument('--mode', type=str, default='phase1_quick',
                        choices=['quick', 'full', 'phase1_quick', 'phase1_full',
                                 'channel', 'sensitivity', 'loss', 'diw'],
                        help='Ablation study mode')
    parser.add_argument('--save_dir', type=str, default='./phase1_results',
                        help='Results save directory')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum training epochs per experiment')

    args = parser.parse_args()

    config = Config()

    if args.mode == 'phase1_quick':
        run_phase1_quick_ablation(config, args.save_dir)
    elif args.mode == 'phase1_full':
        ablation = LightweightAblationStudy(config, args.save_dir)
        ablation.run_phase1_comprehensive_study(args.max_epochs)
    elif args.mode == 'loss':
        ablation = LightweightAblationStudy(config, args.save_dir)
        ablation.run_loss_function_ablation()
        ablation.generate_simple_report()
    elif args.mode == 'diw':
        ablation = LightweightAblationStudy(config, args.save_dir)
        ablation.run_diw_module_comparison()
        ablation.generate_simple_report()
    elif args.mode == 'quick':
        from ablation import run_quick_lightweight_ablation
        run_quick_lightweight_ablation(config, args.save_dir)
    elif args.mode == 'full':
        ablation = LightweightAblationStudy(config, args.save_dir)
        ablation.run_full_lightweight_study(args.max_epochs)
    else:
        print(f"Unknown mode: {args.mode}")
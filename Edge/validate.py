import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import argparse

from config import Config
from models.fusion_net import FusionNet
from data.custom_dataset import get_dataloader, preprocess_batch
from utils.metrics import SegmentationMetrics, LightweightModelProfiler, EdgeMetrics
from utils.visualization import efficient_visualize_predictions, compact_attention_visualization


class LightweightModelValidator:
    def __init__(self, config, checkpoint_path, device=None, auto_config=True):
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if auto_config:
            self.config = self._apply_smart_config(config, checkpoint_path)
        else:
            self.config = config

        self.model = self._load_model()

        self.profiler = LightweightModelProfiler(self.model, self.device)
        self.model_info = self.profiler.count_parameters()

        print(f"Lightweight model validator initialized:")
        print(f"  Model: {checkpoint_path}")
        print(f"  Smart config: {'enabled' if auto_config else 'disabled'}")
        print(f"  Parameters: {self.model_info['total_params']:,}")
        print(f"  Model size: {self.model_info['model_size_mb']:.2f} MB")
        print(f"  Device: {self.device}")

    def _apply_smart_config(self, base_config, checkpoint_path):
        config = base_config

        local_data_config = {
            'DATA_ROOT': config.DATA_ROOT,
            'TRAIN_LIST': config.TRAIN_LIST,
            'VAL_LIST': config.VAL_LIST,
            'IMAGE_DIR': config.IMAGE_DIR,
            'LABEL_DIR': config.LABEL_DIR,
            'MODEL_SAVE_DIR': config.MODEL_SAVE_DIR,
            'LOG_DIR': config.LOG_DIR,
            'RESULT_DIR': config.RESULT_DIR,
        }

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                training_config = checkpoint['config']

                path_related_keys = {
                    'DATA_ROOT', 'TRAIN_LIST', 'VAL_LIST', 'IMAGE_DIR', 'LABEL_DIR',
                    'MODEL_SAVE_DIR', 'LOG_DIR', 'RESULT_DIR'
                }

                for key, value in training_config.__dict__.items():
                    if (hasattr(config, key) and
                        not key.startswith('_') and
                        key not in path_related_keys):
                        setattr(config, key, value)

                for key, value in local_data_config.items():
                    setattr(config, key, value)

                print("Config loaded from checkpoint (paths adapted)")
                print(f"  Training server path: {getattr(training_config, 'DATA_ROOT', 'unknown')}")
                print(f"  Local validation path: {config.DATA_ROOT}")
                return config
        except Exception as e:
            print(f"Cannot load config from checkpoint: {type(e).__name__}")

        model_path = checkpoint_path.lower()

        if 'ultra' in model_path:
            print("Ultra model detected, applying corresponding config")
            config.HIGH_FREQ_CHANNELS = 16
            config.SEMANTIC_CHANNELS = 20
            config.FUSION_CHANNELS = 24
            config.USE_DIW_MODULE = False
            config.USE_PROGRESSIVE_REFINEMENT = False
            config.USE_BOUNDARY_LOSS = True
            config.BOUNDARY_LOSS_WEIGHT = 1.5
            config.EDGE_DETECTION_LAYERS = 1

        elif 'efficient' in model_path:
            print("Efficient model detected, applying corresponding config")
            config.HIGH_FREQ_CHANNELS = 20
            config.SEMANTIC_CHANNELS = 24
            config.FUSION_CHANNELS = 32
            config.USE_DIW_MODULE = False
            config.USE_PROGRESSIVE_REFINEMENT = False
            config.USE_BOUNDARY_LOSS = True
            config.BOUNDARY_LOSS_WEIGHT = 1.0
            config.EDGE_DETECTION_LAYERS = 2

        elif 'accurate' in model_path:
            print("Accurate model detected, applying corresponding config")
            config.HIGH_FREQ_CHANNELS = 24
            config.SEMANTIC_CHANNELS = 32
            config.FUSION_CHANNELS = 48
            config.USE_DIW_MODULE = True
            config.USE_SIMPLIFIED_DIW = True
            config.USE_PROGRESSIVE_REFINEMENT = True
            config.USE_BOUNDARY_LOSS = True
            config.BOUNDARY_LOSS_WEIGHT = 1.2
            config.USE_GRADIENT_LOSS = True
            config.GRADIENT_LOSS_WEIGHT = 0.5
            config.EDGE_DETECTION_LAYERS = 2

        else:
            print("Model type not recognized, using original config")

        return config

    def _load_model(self):
        model = FusionNet(self.config).to(self.device)

        try:
            checkpoint = None
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            except Exception:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Loading model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                state_dict = checkpoint

            try:
                model.load_state_dict(state_dict, strict=True)
                print("Model weights loaded completely")
            except RuntimeError as e:
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() if
                               k in model_dict and v.shape == model_dict[k].shape}

                missing_keys = set(model_dict.keys()) - set(filtered_dict.keys())
                unexpected_keys = set(state_dict.keys()) - set(model_dict.keys())

                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")

                model.load_state_dict(filtered_dict, strict=False)
                print("Compatibility loading completed")

        except Exception as e:
            print(f"Model loading failed: {e}")
            raise e

        model.eval()
        return model

    def benchmark_inference_speed(self, input_shape=None, num_runs=100):
        if input_shape is None:
            input_shape = (1, 3, self.config.INPUT_SIZE, self.config.INPUT_SIZE)
        return self.profiler.measure_inference_speed(input_shape, num_runs)

    def measure_gpu_memory(self, input_shape=None):
        if input_shape is None:
            input_shape = (1, 3, self.config.INPUT_SIZE, self.config.INPUT_SIZE)
        return self.profiler.measure_memory_usage(input_shape)

    def get_comprehensive_profile(self):
        return self.profiler.get_comprehensive_profile()


def validate_model(config, checkpoint_path, result_dir=None, visualize=False,
                  experiment_name=None, auto_config=True):
    if result_dir is None:
        result_dir = config.RESULT_DIR if hasattr(config, 'RESULT_DIR') else "./results"
    if experiment_name is None:
        experiment_name = f"validation_{os.path.basename(checkpoint_path).split('.')[0]}"

    save_dir = os.path.join(result_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting model validation: {experiment_name}")
    print(f"Results save directory: {save_dir}")

    validator = LightweightModelValidator(config, checkpoint_path, auto_config=auto_config)

    print(f"\nRunning performance benchmark tests...")
    benchmark_results = validator.benchmark_inference_speed()
    memory_results = validator.measure_gpu_memory()

    print(f"Inference performance:")
    print(f"  Average inference time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
    print(f"  FPS: {benchmark_results['fps']:.1f}")
    print(f"  GPU memory: {memory_results.get('gpu_memory_peak_mb', 0):.1f} MB")

    print(f"\nStarting dataset validation...")
    val_loader = get_dataloader(validator.config, 'val')

    metrics = SegmentationMetrics(num_classes=validator.config.NUM_CLASSES)
    edge_metrics = None
    if getattr(validator.config, 'EVALUATE_EDGES', False):
        edge_metrics = EdgeMetrics()

    validation_start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
            batch = preprocess_batch(batch, validator.device)

            images = batch['image']
            labels = batch['label']

            if hasattr(validator.model, 'diw_module') and validator.model.diw_module is not None:
                logits, features = validator.model(images, return_features=True)
            else:
                logits = validator.model(images)
                features = None

            preds = torch.argmax(logits, dim=1)

            metrics.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

            if edge_metrics is not None:
                if 'edge_weight' in batch:
                    edge_mask = batch['edge_weight'].cpu().numpy() > 1.5
                else:
                    edge_mask = _generate_edge_mask(labels.cpu().numpy())

                if np.any(edge_mask):
                    edge_metrics.add_batch(
                        preds.cpu().numpy(),
                        labels.cpu().numpy(),
                        edge_mask
                    )

            if visualize and i < 5:
                try:
                    filename = batch['filename'][0] if 'filename' in batch else f"sample_{i}"

                    efficient_visualize_predictions(
                        images.cpu(), labels.cpu(), preds.cpu(),
                        filename, save_dir
                    )

                    if features is not None and 'hf_weight' in features:
                        hf_weight = features['hf_weight'][0].cpu()
                        sem_weight = features['sem_weight'][0].cpu()

                        if hf_weight.dim() == 3 and hf_weight.size(0) == 1:
                            hf_weight = hf_weight.squeeze(0)
                            sem_weight = sem_weight.squeeze(0)

                        weights_combined = torch.stack([hf_weight, sem_weight], dim=-1)
                        compact_attention_visualization(weights_combined, save_dir, filename)

                except Exception as e:
                    print(f"Visualization failed {filename}: {e}")

    validation_time = time.time() - validation_start_time

    results = metrics.get_results()

    if edge_metrics is not None:
        edge_results = edge_metrics.get_metrics()
        for key, value in edge_results.items():
            results[f"Edge_{key}"] = value

    results.update({
        'Performance_FPS': benchmark_results['fps'],
        'Performance_InferenceTime_ms': benchmark_results['avg_inference_time_ms'],
        'Performance_GPUMemory_MB': memory_results.get('gpu_memory_peak_mb', 0),
        'Model_Params': validator.model_info['total_params'],
        'Model_Size_MB': validator.model_info['model_size_mb'],
        'Validation_Time_s': validation_time
    })

    print(f"\nValidation results:")
    print(f"Accuracy metrics:")
    accuracy_metrics = ['Pixel Acc', 'Mean IoU', 'F1 Score', 'Water F1']
    for metric in accuracy_metrics:
        if metric in results:
            print(f"  {metric}: {results[metric]:.4f}")

    print(f"\nPerformance metrics:")
    print(f"  FPS: {results['Performance_FPS']:.1f}")
    print(f"  Inference time: {results['Performance_InferenceTime_ms']:.2f} ms")
    print(f"  GPU memory: {results['Performance_GPUMemory_MB']:.1f} MB")
    print(f"  Parameters: {results['Model_Params']:,}")

    if 'Edge_Precision' in results:
        print(f"\nEdge performance:")
        print(f"  Edge precision: {results['Edge_Precision']:.4f}")
        print(f"  Edge recall: {results['Edge_Recall']:.4f}")
        print(f"  Edge F1: {results['Edge_F1']:.4f}")

    with open(os.path.join(save_dir, 'validation_results.json'), 'w') as f:
        json_results = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=4)

    return results


def validate_multi_models(config, checkpoint_paths, result_dir=None, visualize=False):
    if result_dir is None:
        result_dir = config.RESULT_DIR if hasattr(config, 'RESULT_DIR') else "./results"

    save_dir = os.path.join(result_dir, 'model_comparison')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting multi-model comparison validation")
    print(f"Number of models: {len(checkpoint_paths)}")

    results_dict = {}

    for name, checkpoint_path in checkpoint_paths.items():
        print(f"\n{'=' * 50}")
        print(f"Validating model: {name}")
        print(f"{'=' * 50}")

        try:
            model_save_dir = os.path.join(save_dir, name)
            results = validate_model(
                config, checkpoint_path,
                result_dir=model_save_dir,
                visualize=visualize,
                experiment_name=name,
                auto_config=True
            )
            results_dict[name] = results

        except Exception as e:
            print(f"Model {name} validation failed: {e}")
            results_dict[name] = {'error': str(e)}

    if len(results_dict) > 1:
        _generate_comparison_report(results_dict, save_dir)

    print(f"\nMulti-model validation completed! Comparison results saved to: {save_dir}")

    return results_dict


def validate_optimal_models(base_config=None, result_dir=None, visualize=False):
    if base_config is None:
        base_config = Config()

    models_config = {
        'Ours-Ultra': 'checkpoints/final_ours_ultra/best_model.pth',
        'Ours-Efficient': 'checkpoints/final_ours_efficient/best_model.pth',
        'Ours-Accurate': 'checkpoints/final_ours_accurate/best_model.pth'
    }

    print("Validating all optimal models...")

    results_dict = {}

    for model_name, checkpoint_path in models_config.items():
        if not os.path.exists(checkpoint_path):
            print(f"Model file does not exist: {checkpoint_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Validating model: {model_name}")
        print(f"{'='*50}")

        try:
            results = validate_model(
                config=base_config,
                checkpoint_path=checkpoint_path,
                result_dir=result_dir,
                visualize=visualize,
                experiment_name=f"final_{model_name.lower().replace('-', '_')}",
                auto_config=True
            )

            results_dict[model_name] = results

            print(f"{model_name} validation completed:")
            print(f"  Mean IoU: {results.get('Mean IoU', 0):.4f}")
            print(f"  FPS: {results.get('Performance_FPS', 0):.1f}")
            print(f"  Params: {results.get('Model_Params', 0):,}")

        except Exception as e:
            print(f"{model_name} validation failed: {e}")
            results_dict[model_name] = {'error': str(e)}

    if result_dir:
        summary_dir = os.path.join(result_dir, "optimal_models_summary")
        os.makedirs(summary_dir, exist_ok=True)
        _generate_comparison_report(results_dict, summary_dir)

    return results_dict


def _generate_edge_mask(labels):
    edge_masks = []

    for label in labels:
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        label = label.astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(label, kernel, iterations=1)
        eroded = cv2.erode(label, kernel, iterations=1)
        edge_mask = (dilated - eroded) > 0

        edge_masks.append(edge_mask)

    return np.array(edge_masks)


def _generate_comparison_report(results_dict, save_dir):
    valid_results = {k: v for k, v in results_dict.items() if 'error' not in v}

    if len(valid_results) < 2:
        print("Insufficient valid results, skipping comparison report generation")
        return

    report_lines = ["# Lightweight Model Comparison Report\n"]
    report_lines.append("## Model Performance Comparison\n")

    header = "| Model | mIoU | F1 | Water F1 | FPS | Params(K) | GPU Memory(MB) | Efficiency Score* |"
    separator = "|-------|------|----|---------|----|-----------|----------------|------------------|"
    report_lines.extend([header, separator])

    for model_name, results in valid_results.items():
        miou = results.get('Mean IoU', 0)
        f1 = results.get('F1 Score', 0)
        water_f1 = results.get('Water F1', 0)
        fps = results.get('Performance_FPS', 0)
        params = results.get('Model_Params', 0) / 1000
        gpu_mem = results.get('Performance_GPUMemory_MB', 0)

        efficiency = (miou * fps) / (params * gpu_mem) if params > 0 and gpu_mem > 0 else 0

        row = f"| {model_name} | {miou:.4f} | {f1:.4f} | {water_f1:.4f} | {fps:.1f} | {params:.1f} | {gpu_mem:.1f} | {efficiency:.2f} |"
        report_lines.append(row)

    report_lines.append(f"\n*Efficiency Score = (mIoU × FPS) / (Params K × GPU Memory MB)\n")

    best_miou = max(valid_results.items(), key=lambda x: x[1].get('Mean IoU', 0))
    best_fps = max(valid_results.items(), key=lambda x: x[1].get('Performance_FPS', 0))
    best_params = min(valid_results.items(), key=lambda x: x[1].get('Model_Params', float('inf')))

    report_lines.append(f"## Best Models\n")
    report_lines.append(f"- **Best Accuracy**: {best_miou[0]} (mIoU: {best_miou[1].get('Mean IoU', 0):.4f})")
    report_lines.append(f"- **Best Speed**: {best_fps[0]} (FPS: {best_fps[1].get('Performance_FPS', 0):.1f})")
    report_lines.append(f"- **Fewest Parameters**: {best_params[0]} (Params: {best_params[1].get('Model_Params', 0) / 1000:.1f}K)")

    report_content = "\n".join(report_lines)
    report_path = os.path.join(save_dir, 'comparison_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Comparison report saved: {report_path}")

    try:
        _generate_comparison_plots(valid_results, save_dir)
    except Exception as e:
        print(f"Failed to generate comparison plots: {e}")


def _generate_comparison_plots(results_dict, save_dir):
    import matplotlib.pyplot as plt

    models = list(results_dict.keys())
    mious = [results_dict[m].get('Mean IoU', 0) for m in models]
    fps_values = [results_dict[m].get('Performance_FPS', 0) for m in models]
    params = [results_dict[m].get('Model_Params', 0) / 1000 for m in models]
    gpu_mem = [results_dict[m].get('Performance_GPUMemory_MB', 0) for m in models]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Lightweight Model Performance Comparison', fontsize=16)

    bars1 = ax1.bar(models, mious, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy Comparison (mIoU)')
    ax1.set_ylabel('mIoU')
    ax1.tick_params(axis='x', rotation=45)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom')

    bars2 = ax2.bar(models, fps_values, color='orange', alpha=0.7)
    ax2.set_title('Speed Comparison (FPS)')
    ax2.set_ylabel('FPS')
    ax2.tick_params(axis='x', rotation=45)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom')

    bars3 = ax3.bar(models, params, color='green', alpha=0.7)
    ax3.set_title('Model Size Comparison')
    ax3.set_ylabel('Parameters (K)')
    ax3.tick_params(axis='x', rotation=45)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}K', ha='center', va='bottom')

    efficiency = [(m * f) / (p * g) if p > 0 and g > 0 else 0
                  for m, f, p, g in zip(mious, fps_values, params, gpu_mem)]

    scatter = ax4.scatter(fps_values, mious, s=[p * 20 for p in params],
                          c=efficiency, cmap='viridis', alpha=0.7)
    ax4.set_xlabel('FPS')
    ax4.set_ylabel('mIoU')
    ax4.set_title('Efficiency Trade-off (Bubble size=Parameters, Color=Efficiency)')

    for i, model in enumerate(models):
        ax4.annotate(model, (fps_values[i], mious[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.colorbar(scatter, ax=ax4, label='Efficiency Score')
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plots saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Lightweight model validation tool')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path or directory')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Results save directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization results')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models')
    parser.add_argument('--evaluate_edges', action='store_true',
                        help='Evaluate edge region performance')
    parser.add_argument('--no_auto_config', action='store_true',
                        help='Disable smart configuration detection')
    parser.add_argument('--validate_all_optimal', action='store_true',
                        help='Validate all optimal models')

    args = parser.parse_args()

    config = Config()
    config.EVALUATE_EDGES = args.evaluate_edges

    auto_config = not args.no_auto_config
    print(f"Smart configuration detection: {'enabled' if auto_config else 'disabled'}")

    if args.validate_all_optimal:
        validate_optimal_models(config, args.result_dir, args.visualize)
    elif args.compare and os.path.isdir(args.checkpoint):
        checkpoint_paths = {}
        for filename in os.listdir(args.checkpoint):
            if filename.endswith('.pth'):
                name = os.path.splitext(filename)[0]
                checkpoint_paths[name] = os.path.join(args.checkpoint, filename)

        if not checkpoint_paths:
            raise ValueError(f"No .pth files found in {args.checkpoint}")

        validate_multi_models(config, checkpoint_paths, args.result_dir, args.visualize)
    else:
        if not os.path.exists(args.checkpoint):
            print(f"Model file does not exist: {args.checkpoint}")
            return

        validate_model(config, args.checkpoint, args.result_dir, args.visualize, auto_config=auto_config)


if __name__ == "__main__":
    main()
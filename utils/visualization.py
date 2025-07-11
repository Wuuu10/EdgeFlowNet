import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cv2
import matplotlib
import platform


def setup_chinese_font():
    try:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    except:
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
        return False


has_chinese_font = setup_chinese_font()


def get_label(zh_text, en_text):
    if has_chinese_font:
        return zh_text
    else:
        return en_text


def create_custom_colormap():
    colors = [(0.9, 0.9, 0.9), (0.2, 0.6, 0.9)]
    return LinearSegmentedColormap.from_list('lightweight_water_cmap', colors, N=2)


def create_edge_overlay(label, method='canny'):
    label_uint8 = label.astype(np.uint8)

    if method == 'canny':
        edges = cv2.Canny(label_uint8 * 255, 50, 150)
    else:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(label_uint8, kernel, iterations=1)
        eroded = cv2.erode(label_uint8, kernel, iterations=1)
        edges = dilated - eroded
        edges = edges * 255

    return edges > 0


def efficient_visualize_predictions(images, labels, predictions, filename, save_dir,
                                    create_overlay=True, save_individual=False):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if len(images.shape) == 4:
        image = images[0].transpose(1, 2, 0)
        label = labels[0] if len(labels.shape) > 2 else labels
        pred = predictions[0] if len(predictions.shape) > 2 else predictions
    else:
        image = images.transpose(1, 2, 0)
        label = labels
        pred = predictions

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    water_cmap = create_custom_colormap()

    if create_overlay:
        correct_prediction = (label == pred)
        false_positives = (label == 0) & (pred == 1)
        false_negatives = (label == 1) & (pred == 0)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title(get_label('原始图像', 'Original Image'), fontsize=10)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(label, cmap=water_cmap, vmin=0, vmax=1)
        axes[0, 1].set_title(get_label('真实标签', 'Ground Truth'), fontsize=10)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(pred, cmap=water_cmap, vmin=0, vmax=1)
        axes[1, 0].set_title(get_label('预测结果', 'Prediction'), fontsize=10)
        axes[1, 0].axis('off')

        error_map = np.zeros((*label.shape, 3))
        error_map[correct_prediction] = [0.0, 0.8, 0.0]
        error_map[false_positives] = [0.8, 0.0, 0.0]
        error_map[false_negatives] = [0.0, 0.0, 0.8]

        axes[1, 1].imshow(error_map)

        pixel_accuracy = np.mean(correct_prediction)
        intersection = np.logical_and(pred == 1, label == 1).sum()
        union = np.logical_or(pred == 1, label == 1).sum()
        iou = intersection / union if union > 0 else 0

        axes[1, 1].set_title(f'Error Map (Acc: {pixel_accuracy:.3f}, IoU: {iou:.3f})', fontsize=9)
        axes[1, 1].axis('off')

        base_filename = os.path.basename(filename) if isinstance(filename, str) else f"sample_{filename}"
        fig.suptitle(f'{base_filename}', fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{base_filename.split('.')[0]}_compact.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    if save_individual:
        base_filename = os.path.basename(filename) if isinstance(filename, str) else f"sample_{filename}"

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{base_filename}_original.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(pred, cmap=water_cmap, vmin=0, vmax=1)
        plt.title('Prediction')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{base_filename}_prediction.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    return save_path if create_overlay else None


def visualize_predictions(images, labels, predictions, filename, save_dir):
    return efficient_visualize_predictions(images, labels, predictions, filename, save_dir)


def lightweight_feature_visualization(features, save_dir, filename, max_channels=16):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    n_features = min(features.shape[0], max_channels)
    grid_size = int(np.ceil(np.sqrt(n_features)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    features_min = features.min()
    features_max = features.max()
    normalized_features = (features - features_min) / (features_max - features_min + 1e-8)

    for i in range(n_features):
        row, col = i // grid_size, i % grid_size
        ax = axes[row, col]
        feature = normalized_features[i]
        ax.imshow(feature, cmap='viridis')
        ax.set_title(f'Ch {i}', fontsize=8)
        ax.axis('off')

    for i in range(n_features, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        axes[row, col].axis('off')

    fig.suptitle(f'Feature Maps - {filename}', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{filename}_features.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return save_path


def visualize_feature_maps(features, save_dir, filename):
    return lightweight_feature_visualization(features, save_dir, filename)


def compact_attention_visualization(attention_weights, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    if len(attention_weights.shape) == 4:
        attention_weights = attention_weights[0].transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    high_freq_weight = attention_weights[..., 0]
    im1 = axes[0].imshow(high_freq_weight, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title('High-Freq Weight', fontsize=10)
    axes[0].axis('off')

    semantic_weight = attention_weights[..., 1]
    im2 = axes[1].imshow(semantic_weight, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Semantic Weight', fontsize=10)
    axes[1].axis('off')

    fig.suptitle(f'DIW Weights - {filename}', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{filename}_weights.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return save_path


def visualize_attention_weights(attention_weights, save_dir, filename):
    return compact_attention_visualization(attention_weights, save_dir, filename)


def lightweight_model_comparison(images, labels, predictions_dict, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if len(images.shape) == 4:
        image = images[0].transpose(1, 2, 0)
        label = labels[0] if len(labels.shape) > 2 else labels
    else:
        image = images.transpose(1, 2, 0)
        label = labels

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    water_cmap = create_custom_colormap()

    n_models = len(predictions_dict)
    n_cols = min(3, n_models + 1)
    n_rows = (n_models + 2) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    if n_cols > 1:
        axes[0, 1].imshow(label, cmap=water_cmap, vmin=0, vmax=1)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')

    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if len(predictions.shape) > 2:
            pred = predictions[0]
        else:
            pred = predictions

        row = (i + 2) // n_cols
        col = (i + 2) % n_cols

        if row < n_rows and col < n_cols:
            axes[row, col].imshow(pred, cmap=water_cmap, vmin=0, vmax=1)

            pixel_accuracy = np.mean(label == pred)
            intersection = np.logical_and(pred == 1, label == 1).sum()
            union = np.logical_or(pred == 1, label == 1).sum()
            iou = intersection / union if union > 0 else 0

            axes[row, col].set_title(f'{model_name}\nAcc: {pixel_accuracy:.3f}, IoU: {iou:.3f}', fontsize=9)
            axes[row, col].axis('off')

    for i in range(len(predictions_dict) + 2, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if row < n_rows and col < n_cols:
            axes[row, col].axis('off')

    base_filename = os.path.basename(filename) if isinstance(filename, str) else f"sample_{filename}"
    fig.suptitle(f'Model Comparison: {base_filename}', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{base_filename}_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def compare_models_visualization(images, labels, predictions_dict, save_dir, filename):
    return lightweight_model_comparison(images, labels, predictions_dict, save_dir, filename)


def create_efficiency_plot(model_results, save_dir, plot_name="efficiency_analysis"):
    os.makedirs(save_dir, exist_ok=True)

    models = list(model_results.keys())
    mious = [results.get('Mean IoU', 0) for results in model_results.values()]
    fps_values = [results.get('Performance_FPS', 0) for results in model_results.values()]
    params = [results.get('Model_Params', 0) / 1e6 for results in model_results.values()]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.bar(models, mious, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('mIoU')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(models, fps_values, color='orange', alpha=0.7)
    ax2.set_title('Speed Comparison')
    ax2.set_ylabel('FPS')
    ax2.tick_params(axis='x', rotation=45)

    ax3.bar(models, params, color='green', alpha=0.7)
    ax3.set_title('Model Size Comparison')
    ax3.set_ylabel('Parameters (M)')
    ax3.tick_params(axis='x', rotation=45)

    scatter = ax4.scatter(fps_values, mious, s=[p * 100 for p in params],
                          c=range(len(models)), cmap='viridis', alpha=0.7)
    ax4.set_xlabel('FPS')
    ax4.set_ylabel('mIoU')
    ax4.set_title('Efficiency Trade-off\n(Bubble size = Parameters)')

    for i, model in enumerate(models):
        ax4.annotate(model, (fps_values[i], mious[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return save_path


def create_parameter_distribution_plot(model, save_dir, model_name="model"):
    os.makedirs(save_dir, exist_ok=True)

    module_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                module_params[name] = params

    if not module_params:
        return None

    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    top_modules = sorted_modules[:10]

    names = [item[0].split('.')[-1] for item in top_modules]
    params = [item[1] / 1000 for item in top_modules]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), params, color='lightcoral', alpha=0.7)
    plt.title(f'Parameter Distribution - {model_name}')
    plt.ylabel('Parameters (K)')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.1f}K', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_param_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def visualize_training_curves(train_losses, val_losses, val_mious, save_dir, experiment_name="training"):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_mious, 'g-', label='Val mIoU', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Validation mIoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{experiment_name}_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def create_ablation_summary_plot(ablation_results, save_dir, study_name="ablation"):
    os.makedirs(save_dir, exist_ok=True)

    variants = []
    mious = []
    fps_values = []
    params = []

    for variant_name, result in ablation_results.items():
        if 'error' in result:
            continue

        variants.append(variant_name.replace('_', '\n'))
        mious.append(result.get('best_miou', 0))
        fps_values.append(result['performance']['fps'])
        params.append(result['model_info']['total_params'] / 1e6)

    if not variants:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].barh(range(len(variants)), mious, color='skyblue', alpha=0.7)
    axes[0, 0].set_yticks(range(len(variants)))
    axes[0, 0].set_yticklabels(variants, fontsize=8)
    axes[0, 0].set_xlabel('mIoU')
    axes[0, 0].set_title('Accuracy Comparison')

    axes[0, 1].barh(range(len(variants)), fps_values, color='orange', alpha=0.7)
    axes[0, 1].set_yticks(range(len(variants)))
    axes[0, 1].set_yticklabels(variants, fontsize=8)
    axes[0, 1].set_xlabel('FPS')
    axes[0, 1].set_title('Speed Comparison')

    axes[1, 0].barh(range(len(variants)), params, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(variants)))
    axes[1, 0].set_yticklabels(variants, fontsize=8)
    axes[1, 0].set_xlabel('Parameters (M)')
    axes[1, 0].set_title('Model Size Comparison')

    efficiency = [(m * f) / p if p > 0 else 0 for m, f, p in zip(mious, fps_values, params)]
    axes[1, 1].barh(range(len(variants)), efficiency, color='red', alpha=0.7)
    axes[1, 1].set_yticks(range(len(variants)))
    axes[1, 1].set_yticklabels(variants, fontsize=8)
    axes[1, 1].set_xlabel('Efficiency Score')
    axes[1, 1].set_title('Overall Efficiency')

    fig.suptitle(f'Ablation Study Results: {study_name}', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{study_name}_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return save_path
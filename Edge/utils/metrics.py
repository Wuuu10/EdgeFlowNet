import numpy as np
import torch
import time
import psutil
import os


class SegmentationMetrics:
    """分割评估指标计算器"""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_true, label_pred):
        """快速计算混淆矩阵"""
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, label_true, label_pred):
        """添加一个批次的预测结果"""
        assert label_true.shape == label_pred.shape
        self.confusion_matrix += self._fast_hist(label_true.flatten(), label_pred.flatten())

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_results(self):
        """计算并返回评估指标"""
        if self.confusion_matrix.sum() == 0:
            return self._get_zero_metrics()

        # 像素准确率
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        # 类别准确率
        acc_cls = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        acc_cls = np.nanmean(acc_cls)

        # IoU计算
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) -
                 np.diag(self.confusion_matrix))

        iou = intersection / (union + 1e-10)
        valid_iou = iou[union > 0]
        miou = np.mean(valid_iou) if len(valid_iou) > 0 else 0.0

        # Frequency Weighted IoU
        freq = self.confusion_matrix.sum(axis=1) / (self.confusion_matrix.sum() + 1e-10)
        fwiou = np.sum(freq * iou)

        # Precision, Recall, F1
        precision = intersection / (self.confusion_matrix.sum(axis=0) + 1e-10)
        recall = intersection / (self.confusion_matrix.sum(axis=1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        # 有效值的平均
        valid_precision = precision[self.confusion_matrix.sum(axis=0) > 0]
        valid_recall = recall[self.confusion_matrix.sum(axis=1) > 0]
        valid_f1 = f1[(self.confusion_matrix.sum(axis=0) > 0) &
                      (self.confusion_matrix.sum(axis=1) > 0)]

        # 水体类别特定指标（假设类别1是水体）
        water_f1 = 0.0
        water_precision = 0.0
        water_recall = 0.0
        if self.num_classes > 1:
            water_precision = precision[1] if len(precision) > 1 else 0.0
            water_recall = recall[1] if len(recall) > 1 else 0.0
            water_f1 = f1[1] if len(f1) > 1 else 0.0

        results = {
            'Pixel Acc': float(acc),
            'Class Acc': float(acc_cls),
            'Mean IoU': float(miou),
            'FW IoU': float(fwiou),
            'F1 Score': float(np.mean(valid_f1)) if len(valid_f1) > 0 else 0.0,
            'Precision': float(np.mean(valid_precision)) if len(valid_precision) > 0 else 0.0,
            'Recall': float(np.mean(valid_recall)) if len(valid_recall) > 0 else 0.0,
            'Water F1': float(water_f1),
            'Water Precision': float(water_precision),
            'Water Recall': float(water_recall)
        }

        # 添加各类别IoU
        for i in range(self.num_classes):
            if union[i] > 0:
                results[f'IoU Class {i}'] = float(iou[i])
            else:
                results[f'IoU Class {i}'] = 0.0

        return results

    def _get_zero_metrics(self):
        """返回零值指标"""
        results = {
            'Pixel Acc': 0.0,
            'Class Acc': 0.0,
            'Mean IoU': 0.0,
            'FW IoU': 0.0,
            'F1 Score': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'Water F1': 0.0,
            'Water Precision': 0.0,
            'Water Recall': 0.0
        }

        for i in range(self.num_classes):
            results[f'IoU Class {i}'] = 0.0

        return results


class EdgeMetrics:
    """边缘区域专用评估指标"""

    def __init__(self):
        self.edge_tp = 0
        self.edge_fp = 0
        self.edge_fn = 0
        self.edge_tn = 0

    def add_batch(self, pred, target, edge_mask):
        """添加边缘区域的预测结果"""
        pred_edge = pred[edge_mask]
        target_edge = target[edge_mask]

        self.edge_tp += np.sum((pred_edge == 1) & (target_edge == 1))
        self.edge_fp += np.sum((pred_edge == 1) & (target_edge == 0))
        self.edge_fn += np.sum((pred_edge == 0) & (target_edge == 1))
        self.edge_tn += np.sum((pred_edge == 0) & (target_edge == 0))

    def get_metrics(self):
        """计算边缘区域指标"""
        precision = self.edge_tp / (self.edge_tp + self.edge_fp + 1e-10)
        recall = self.edge_tp / (self.edge_tp + self.edge_fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        accuracy = (self.edge_tp + self.edge_tn) / (
                self.edge_tp + self.edge_fp + self.edge_fn + self.edge_tn + 1e-10
        )

        return {
            'Edge Precision': float(precision),
            'Edge Recall': float(recall),
            'Edge F1': float(f1),
            'Edge Accuracy': float(accuracy)
        }

    def reset(self):
        """重置计数器"""
        self.edge_tp = 0
        self.edge_fp = 0
        self.edge_fn = 0
        self.edge_tn = 0


class LightweightModelProfiler:
    """轻量级模型性能分析器"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def count_parameters(self):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

    def count_flops(self, input_shape=(1, 3, 256, 256)):
        """计算FLOPs (需要thop库)"""
        try:
            from thop import profile
            dummy_input = torch.randn(input_shape).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

            return {
                'flops': flops,
                'gflops': flops / 1e9,
                'mflops': flops / 1e6
            }
        except ImportError:
            return {'flops': 0, 'gflops': 0, 'mflops': 0}
        except Exception as e:
            return {'flops': 0, 'gflops': 0, 'mflops': 0}

    def measure_inference_speed(self, input_shape=(1, 3, 256, 256), num_runs=100):
        """测量推理速度"""
        self.model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        # 同步GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 测量时间
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        total_time = end_time - start_time
        avg_time = (total_time / num_runs) * 1000  # 转换为毫秒
        fps = num_runs / total_time

        return {
            'avg_inference_time_ms': avg_time,
            'fps': fps,
            'throughput_imgs_per_sec': fps
        }

    def measure_memory_usage(self, input_shape=(1, 3, 256, 256)):
        """测量内存使用量"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dummy_input = torch.randn(input_shape).to(self.device)

            with torch.no_grad():
                _ = self.model(dummy_input)

            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)

            return {
                'gpu_memory_peak_mb': peak_memory,
                'gpu_memory_current_mb': current_memory
            }
        else:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'cpu_memory_mb': memory_info.rss / (1024 * 1024),
                'gpu_memory_peak_mb': 0,
                'gpu_memory_current_mb': 0
            }

    def get_comprehensive_profile(self, input_shape=(1, 3, 256, 256)):
        """获取综合性能分析"""
        profile_results = {}

        # 参数统计
        profile_results.update(self.count_parameters())

        # FLOPs统计
        profile_results.update(self.count_flops(input_shape))

        # 速度测试
        profile_results.update(self.measure_inference_speed(input_shape))

        # 内存测试
        profile_results.update(self.measure_memory_usage(input_shape))

        # 效率指标
        efficiency_metrics = self._calculate_efficiency_metrics(profile_results)
        profile_results.update(efficiency_metrics)

        return profile_results

    def _calculate_efficiency_metrics(self, metrics):
        """计算效率指标"""
        params_m = metrics.get('total_params', 0) / 1e6
        flops_g = metrics.get('gflops', 0)
        fps = metrics.get('fps', 0)
        memory_mb = metrics.get('gpu_memory_peak_mb', metrics.get('cpu_memory_mb', 1))

        return {
            'params_efficiency': fps / params_m if params_m > 0 else 0,
            'flops_efficiency': fps / flops_g if flops_g > 0 else 0,
            'memory_efficiency': fps / memory_mb if memory_mb > 0 else 0,
            'overall_efficiency': (fps * 100) / (params_m + flops_g + memory_mb / 100)
            if (params_m + flops_g + memory_mb) > 0 else 0
        }


class ModelComparator:
    """模型比较工具"""

    def __init__(self):
        self.models_results = {}

    def add_model(self, name, results):
        """添加模型结果"""
        self.models_results[name] = results

    def compare_metrics(self, metrics=['Mean IoU', 'fps', 'total_params']):
        """比较指定指标"""
        comparison = {}

        for metric in metrics:
            values = []
            models = []

            for model_name, results in self.models_results.items():
                if metric in results:
                    values.append(results[metric])
                    models.append(model_name)

            if values:
                comparison[metric] = {
                    'models': models,
                    'values': values,
                    'best_model': models[np.argmax(values)] if metric != 'total_params'
                    else models[np.argmin(values)],
                    'best_value': max(values) if metric != 'total_params' else min(values)
                }

        return comparison

    def generate_ranking(self, weights=None):
        """生成模型排名"""
        if weights is None:
            weights = {
                'Mean IoU': 0.4,
                'fps': 0.3,
                'total_params': -0.3  # 负权重，参数越少越好
            }

        rankings = {}

        for model_name, results in self.models_results.items():
            score = 0
            for metric, weight in weights.items():
                if metric in results:
                    value = results[metric]
                    if metric == 'total_params':
                        # 参数量标准化 (越少越好)
                        max_params = max(res.get('total_params', 0) for res in self.models_results.values())
                        normalized_value = 1 - (value / max_params) if max_params > 0 else 1
                    else:
                        # 其他指标标准化 (越大越好)
                        max_value = max(res.get(metric, 0) for res in self.models_results.values())
                        normalized_value = value / max_value if max_value > 0 else 0

                    score += weight * normalized_value

            rankings[model_name] = score

        # 按分数排序
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        return {
            'rankings': sorted_rankings,
            'best_model': sorted_rankings[0][0] if sorted_rankings else None,
            'scores': rankings
        }


class LightweightBenchmark:
    """轻量级模型基准测试"""

    def __init__(self, model, device, input_shape=(1, 3, 256, 256)):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.profiler = LightweightModelProfiler(model, device)

    def run_speed_benchmark(self, batch_sizes=[1, 2, 4, 8]):
        """运行速度基准测试"""
        results = {}

        for bs in batch_sizes:
            try:
                input_shape = (bs, *self.input_shape[1:])
                speed_metrics = self.profiler.measure_inference_speed(input_shape, num_runs=50)
                results[f'batch_size_{bs}'] = speed_metrics
                print(f"bathsize{bs}: {speed_metrics['fps']:.1f} FPS")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    break
                else:
                    raise e

        return results

    def run_memory_benchmark(self, batch_sizes=[1, 2, 4, 8]):
        """运行内存基准测试"""
        results = {}

        for bs in batch_sizes:
            try:
                input_shape = (bs, *self.input_shape[1:])
                memory_metrics = self.profiler.measure_memory_usage(input_shape)
                results[f'batch_size_{bs}'] = memory_metrics
                print(f"bathsize{bs}: {memory_metrics.get('gpu_memory_peak_mb', 0):.1f} MB")

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    break
                else:
                    raise e

        return results

    def get_efficiency_summary(self):
        """获取效率摘要"""
        summary = self.profiler.get_comprehensive_profile(self.input_shape)

        # 添加轻量化等级评估
        params_m = summary['total_params'] / 1e6
        gflops = summary['gflops']
        fps = summary['fps']

        # 轻量化等级评分 (0-100)
        param_score = max(0, 100 - params_m * 10)  # 每M参数扣10分
        flops_score = max(0, 100 - gflops * 2)  # 每GFLOP扣2分
        speed_score = min(100, fps * 2)  # 每FPS得2分，最高100分

        lightweight_score = (param_score + flops_score + speed_score) / 3

        summary['lightweight_score'] = lightweight_score
        summary['param_score'] = param_score
        summary['flops_score'] = flops_score
        summary['speed_score'] = speed_score

        return summary

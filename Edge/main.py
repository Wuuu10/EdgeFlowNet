import os
import sys
import argparse
import json
import torch

from config import Config
from train import LightweightTrainer
from validate import validate_model, validate_multi_models, LightweightModelValidator
from ablation import LightweightAblationStudy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='轻量级边缘检测网络 - 主程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练模型
  python main.py --mode train --experiment_name my_experiment

  # 验证模型
  python main.py --mode validate --checkpoint ./checkpoints/best_model.pth

  # 运行消融研究
  python main.py --mode ablation --ablation_mode quick

  # 性能基准测试
  python main.py --mode benchmark --checkpoint ./checkpoints/best_model.pth

  # 多模型比较
  python main.py --mode compare --checkpoint ./checkpoints/ --visualize
        """
    )

    # 主要参数
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'validate', 'test', 'ablation', 'benchmark', 'compare', 'profile'],
                        help='运行模式')

    parser.add_argument('--config', type=str, default=None,
                        help='自定义配置文件路径 (JSON格式)')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径 (文件或目录)')

    parser.add_argument('--experiment_name', type=str, default=None,
                        help='实验名称')

    # 验证和可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化结果')

    parser.add_argument('--save_reports', action='store_true',
                        help='保存详细报告')

    parser.add_argument('--evaluate_edges', action='store_true',
                        help='评估边缘区域性能')

    # 消融研究参数
    parser.add_argument('--ablation_mode', type=str, default='quick',
                        choices=['quick', 'full', 'channel', 'module', 'edge'],
                        help='消融研究模式')

    parser.add_argument('--max_epochs', type=int, default=50,
                        help='消融研究的最大训练轮数')

    # 训练参数
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')

    parser.add_argument('--efficient_mode', action='store_true',
                        help='使用效率优化版训练器')

    # 输出参数
    parser.add_argument('--result_dir', type=str, default=None,
                        help='结果保存目录')

    parser.add_argument('--quiet', action='store_true',
                        help='减少输出信息')

    return parser.parse_args()


def load_custom_config(config_path):
    """加载自定义配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 创建基础配置
    config = Config()

    # 更新配置
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"警告: 未知配置项 {key}")

    print(f"已加载自定义配置: {config_path}")
    return config


def train_mode(config, args):
    """训练模式"""
    print("训练模式")

    # 选择训练器
    if args.efficient_mode:
        trainer_class = LightweightTrainer
        print("使用标准训练器")

    # 创建训练器
    experiment_name = args.experiment_name or config.EXPERIMENT_NAME
    trainer = trainer_class(config, experiment_name)

    # 从检查点恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.resume_training(args.resume)
            config.NUM_EPOCHS = config.NUM_EPOCHS - start_epoch
            print(f"从第 {start_epoch} 轮恢复训练")
        else:
            print(f"警告: 检查点文件不存在 {args.resume}")

    # 开始训练
    best_miou = trainer.train()

    print(f"训练完成")
    print(f"最佳mIoU: {best_miou:.4f}")
    print(f"模型保存路径: {os.path.join(config.MODEL_SAVE_DIR, experiment_name)}")

    return best_miou


def validate_mode(config, args):
    """验证模式"""
    print("启动验证模式")

    if not args.checkpoint:
        raise ValueError("验证模式需要指定 --checkpoint 参数")

    experiment_name = args.experiment_name or f"validation_{os.path.basename(args.checkpoint).split('.')[0]}"

    # 设置边缘评估
    config.EVALUATE_EDGES = args.evaluate_edges

    results = validate_model(
        config=config,
        checkpoint_path=args.checkpoint,
        result_dir=args.result_dir,
        visualize=args.visualize,
        experiment_name=experiment_name
    )

    print("验证完成!")
    return results


def compare_mode(config, args):
    """比较模式"""
    print("启动多模型比较模式")

    if not args.checkpoint:
        raise ValueError("比较模式需要指定 --checkpoint 参数")

    if os.path.isdir(args.checkpoint):
        # 从目录中找所有模型
        checkpoint_paths = {}
        for filename in os.listdir(args.checkpoint):
            if filename.endswith('.pth'):
                name = os.path.splitext(filename)[0]
                checkpoint_paths[name] = os.path.join(args.checkpoint, filename)

        if not checkpoint_paths:
            raise ValueError(f"在目录 {args.checkpoint} 中未找到 .pth 文件")

        print(f"找到 {len(checkpoint_paths)} 个模型文件")
        for name in checkpoint_paths:
            print(f"  - {name}")

    else:
        raise ValueError("比较模式需要指定包含多个模型的目录")

    # 设置边缘评估
    config.EVALUATE_EDGES = args.evaluate_edges

    results = validate_multi_models(
        config=config,
        checkpoint_paths=checkpoint_paths,
        result_dir=args.result_dir,
        visualize=args.visualize
    )

    print("模型比较完成")
    return results


def benchmark_mode(config, args):
    """基准测试模式"""
    print("⚡ 启动性能基准测试模式")

    if not args.checkpoint:
        raise ValueError("基准测试模式需要指定 --checkpoint 参数")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")

    # 创建验证器
    validator = LightweightModelValidator(config, args.checkpoint, device)

    print("\n模型信息:")
    model_info = validator.model_info
    print(f"  总参数量: {model_info['total_params']:,}")
    print(f"  可训练参数: {model_info['trainable_params']:,}")
    print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")

    print("\n速度基准测试:")
    speed_results = validator.benchmark_inference_speed(num_runs=200)
    print(f"  平均推理时间: {speed_results['avg_inference_time_ms']:.2f} ms")
    print(f"  FPS: {speed_results['fps']:.1f}")
    print(f"  吞吐量: {speed_results['throughput_imgs_per_sec']:.1f} images/sec")

    print("\n 内存基准测试:")
    memory_results = validator.measure_gpu_memory()
    if device.type == 'cuda':
        print(f"  GPU内存峰值: {memory_results['gpu_memory_peak_mb']:.1f} MB")
        print(f"  GPU内存当前: {memory_results['gpu_memory_current_mb']:.1f} MB")
    else:
        print(f"  CPU内存: {memory_results['cpu_memory_mb']:.1f} MB")

    # 综合性能分析
    print(f"\n综合性能分析:")
    comprehensive_results = validator.get_comprehensive_profile()

    efficiency_metrics = ['params_efficiency', 'flops_efficiency', 'memory_efficiency', 'overall_efficiency']
    for metric in efficiency_metrics:
        if metric in comprehensive_results:
            print(f"  {metric.replace('_', ' ').title()}: {comprehensive_results[metric]:.2f}")

    # 保存报告
    if args.save_reports:
        result_dir = args.result_dir or config.RESULT_DIR
        experiment_name = args.experiment_name or "benchmark_results"
        save_dir = os.path.join(result_dir, experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        # 合并所有结果
        all_results = {**model_info, **speed_results, **memory_results, **comprehensive_results}

        # 保存JSON结果
        with open(os.path.join(save_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

        # 生成效率报告
        from utils.metrics import generate_efficiency_report
        generate_efficiency_report(
            experiment_name,
            all_results,
            os.path.join(save_dir, "efficiency_report.md")
        )

        print(f"\n基准测试报告已保存到: {save_dir}")

    print(" 基准测试完成")
    return comprehensive_results


def ablation_mode(config, args):
    print("启动消融研究模式")
    print(f"消融模式: {args.ablation_mode}")
    print(f"最大训练轮数: {args.max_epochs}")

    result_dir = args.result_dir or os.path.join(config.RESULT_DIR, "ablation_results")

    if args.ablation_mode == 'full':
        ablation = LightweightAblationStudy(config, result_dir)
        ablation.run_full_lightweight_study(args.max_epochs)
        results = ablation.results

    elif args.ablation_mode == 'channel':
        print("运行通道配置消融研究...")
        ablation = LightweightAblationStudy(config, result_dir)
        ablation.run_lightweight_channel_ablation()
        results = ablation.results

    elif args.ablation_mode == 'module':
        print("运行模块消融研究...")
        ablation = LightweightAblationStudy(config, result_dir)
        ablation.run_lightweight_module_ablation()
        results = ablation.results

    elif args.ablation_mode == 'edge':
        print("运行边缘处理消融研究...")
        ablation = LightweightAblationStudy(config, result_dir)
        ablation.run_edge_processing_ablation()
        results = ablation.results

    else:
        raise ValueError(f"不支持的消融模式: {args.ablation_mode}")

    print("消融研究完成")
    return results


def profile_mode(config, args):
    print("启动性能分析模式")
    if not args.checkpoint:
        raise ValueError("性能分析模式需要指定 --checkpoint 参数")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    validator = LightweightModelValidator(config, args.checkpoint, device)
    profile_results = validator.get_comprehensive_profile()

    print("\n性能分析:")
    print(f"模型参数: {profile_results['total_params']:,}")
    print(f"模型大小: {profile_results['model_size_mb']:.2f} MB")
    print(f"计算量: {profile_results['gflops']:.2f} GFLOPs")
    print(f"推理速度: {profile_results['fps']:.1f} FPS")
    print(f"GPU内存: {profile_results.get('gpu_memory_peak_mb', 0):.1f} MB")
    print(f"参数效率: {profile_results['params_efficiency']:.2f}")
    print(f"综合效率: {profile_results['overall_efficiency']:.2f}")

    # 轻量化评估
    from utils.metrics import LightweightBenchmark
    benchmark = LightweightBenchmark(validator.model, device)
    efficiency_summary = benchmark.get_efficiency_summary()

    print(f"\n轻量化评估:")
    print(f"轻量化分数: {efficiency_summary['lightweight_score']:.1f}/100")
    print(f"  参数分数: {efficiency_summary['param_score']:.1f}/100")
    print(f"  计算分数: {efficiency_summary['flops_score']:.1f}/100")
    print(f"  速度分数: {efficiency_summary['speed_score']:.1f}/100")

    # 保存详细分析
    if args.save_reports:
        result_dir = args.result_dir or config.RESULT_DIR
        experiment_name = args.experiment_name or "profile_results"
        save_dir = os.path.join(result_dir, experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        # 保存性能分析结果
        all_results = {**profile_results, **efficiency_summary}
        with open(os.path.join(save_dir, 'profile_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

        # 生成参数分布图
        try:
            from utils.visualization import create_parameter_distribution_plot
            create_parameter_distribution_plot(validator.model, save_dir, experiment_name)
        except Exception as e:
            print(f"参数分布图生成失败: {e}")

        print(f"\n性能分析报告已保存到: {save_dir}")

    print("性能分析完成")
    return profile_results


def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    if not args.quiet:
        print(f"运行模式: {args.mode}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU设备: {torch.cuda.get_device_name()}")

    # 加载配置
    if args.config:
        config = load_custom_config(args.config)
    else:
        config = Config()

    # 根据参数更新配置
    if args.result_dir:
        config.RESULT_DIR = args.result_dir

    # 显示配置摘要
    if not args.quiet:
        print(f"\n配置摘要:")
        print(f"  数据根目录: {config.DATA_ROOT}")
        print(f"  输入尺寸: {config.INPUT_SIZE}")
        print(f"  批次大小: {config.BATCH_SIZE}")
        print(f"  结果目录: {config.RESULT_DIR}")
        print(
            f"  模型通道: HF={config.HIGH_FREQ_CHANNELS}, Sem={config.SEMANTIC_CHANNELS}, Fus={config.FUSION_CHANNELS}")

    try:
        # 执行相应模式
        if args.mode == 'train':
            result = train_mode(config, args)

        elif args.mode == 'validate':
            result = validate_mode(config, args)

        elif args.mode == 'compare':
            result = compare_mode(config, args)

        elif args.mode == 'benchmark':
            result = benchmark_mode(config, args)

        elif args.mode == 'ablation':
            result = ablation_mode(config, args)

        elif args.mode == 'profile':
            result = profile_mode(config, args)

        else:
            raise ValueError(f"不支持的运行模式: {args.mode}")

        if not args.quiet:
            print(f"\n🎉 {args.mode.upper()} 模式执行完成!")

        return result

    except KeyboardInterrupt:
        return None

    except Exception as e:
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置环境变量
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

    # 运行主程序
    result = main()

    # 退出程序
    if result is not None:
        sys.exit(0)
    else:
        sys.exit(1)
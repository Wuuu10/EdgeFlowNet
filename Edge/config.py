import os


class Config:
    DATA_ROOT = "/root/autodl-fs/dataset"
    TRAIN_LIST = "train.txt"
    VAL_LIST = "val.txt"
    IMAGE_DIR = "image"
    LABEL_DIR = "SegmentationClassPNG"

    NUM_CLASSES = 2
    WATER_LABEL = 8
    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 150

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'AdamW'
    SCHEDULER = 'CosineAnnealingLR'
    MIN_LR = 1e-6

    GRADIENT_ACCUMULATION_STEPS = 8
    GRADIENT_CLIP_NORM = 0.5
    MIXED_PRECISION = True  # 启用混合精度以节省内存

    # 早停配置
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_DELTA = 0.001

    # ====================== 输入数据配置 ======================
    INPUT_SIZE = 256  # 保持256尺寸，但优化内存使用
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ====================== 轻量化网络架构配置 ======================
    # 双分支通道配置 - 核心创新点
    HIGH_FREQ_CHANNELS = 24  # 高频分支通道数
    SEMANTIC_CHANNELS = 32  # 语义分支通道数
    FUSION_CHANNELS = 48  # 融合特征通道数

    # ====================== Phase 1: 极致轻量化配置 ======================
    # 超轻量化通道配置变体
    ULTRA_LIGHT_V2_HIGH_FREQ_CHANNELS = 14  # 从16进一步减少到14
    ULTRA_LIGHT_V2_SEMANTIC_CHANNELS = 18  # 从20减少到18
    ULTRA_LIGHT_V2_FUSION_CHANNELS = 22  # 从24减少到22

    ULTRA_LIGHT_V3_HIGH_FREQ_CHANNELS = 12  # 更激进的配置
    ULTRA_LIGHT_V3_SEMANTIC_CHANNELS = 16
    ULTRA_LIGHT_V3_FUSION_CHANNELS = 20

    # 最小配置 - 极限测试
    MINIMAL_HIGH_FREQ_CHANNELS = 10
    MINIMAL_SEMANTIC_CHANNELS = 14
    MINIMAL_FUSION_CHANNELS = 18

    # DIW模块配置 - 创新点：动态重要性权重
    USE_DIW_MODULE = True
    DIW_REDUCTION_RATIO = 8  # DIW模块中的通道缩减比例
    DIW_ATTENTION_DIM = 8  # 注意力计算的维度

    # ====================== Phase 1: DIW模块类型选择 ======================
    DIW_MODULE_TYPE = 'standard'  # 'standard', 'ultra_lightweight', 'simplified'
    USE_SIMPLIFIED_DIW = False  # 新增：是否使用简化版DIW模块

    # 边缘增强配置 - 创新点：边缘感知
    EDGE_ENHANCEMENT_STRENGTH = 2.0  # 边缘增强强度
    EDGE_DETECTION_LAYERS = 2  # 边缘检测层数

    # ====================== Phase 1: 边缘检测优化 ======================
    EDGE_DETECTION_LAYERS_MINIMAL = 1  # 从2减少到1
    EDGE_DETECTION_LAYERS_ULTRA_MINIMAL = 0  # 完全移除边缘检测层

    # 激活函数优化选项
    USE_SIMPLE_ACTIVATION = True  # 使用简单ReLU替代复杂激活

    # 轻量化模块开关
    USE_PROGRESSIVE_REFINEMENT = True  # 渐进细化模块
    USE_CHANNEL_ATTENTION = True  # 通道注意力
    USE_SPATIAL_ATTENTION = True  # 空间注意力
    USE_RESIDUAL_CONNECTIONS = False  # 新增：是否使用残差连接

    # ====================== 损失函数配置 ======================
    # 边界损失 - 强化边缘检测
    USE_BOUNDARY_LOSS = True
    BOUNDARY_LOSS_WEIGHT = 3.0

    # ====================== Phase 1: 优化后的损失函数配置 ======================
    # 稳定的边界损失权重
    BOUNDARY_LOSS_WEIGHT_STABLE = 1.5  # 从3.0降到1.5
    BOUNDARY_LOSS_WEIGHT_MINIMAL = 1.0  # 最小权重测试
    BOUNDARY_LOSS_WEIGHT_ULTRA_MINIMAL = 0.5  # 超最小权重

    # 损失函数类型选择
    LOSS_TYPE = 'combined'  # 'combined', 'stable', 'simplified', 'weighted'

    # 简化损失函数选项
    USE_SIMPLIFIED_LOSS = True  # 只使用CE + Dice
    USE_GRADIENT_LOSS_MINIMAL = True  # 最小梯度损失

    # 梯度一致性损失
    USE_GRADIENT_LOSS = True
    GRADIENT_LOSS_WEIGHT = 1.0
    GRADIENT_LOSS_WEIGHT_MINIMAL = 0.5  # 从1.0降到0.5
    GRADIENT_LOSS_WEIGHT_ULTRA_MINIMAL = 0.2  # 超最小权重

    # 对比损失权重
    CONTRASTIVE_LOSS_WEIGHT = 0.3
    CONTRASTIVE_LOSS_WEIGHT_MINIMAL = 0.1  # 降低权重
    EDGE_FOCAL_LOSS_WEIGHT = 0.5
    EDGE_FOCAL_LOSS_WEIGHT_MINIMAL = 0.2  # 降低权重

    # ====================== 数据加载优化配置 ======================
    NUM_WORKERS = 0  # Windows兼容性
    PIN_MEMORY = False  # 与NUM_WORKERS=0保持一致
    PREFETCH_FACTOR = 2

    # ====================== 内存优化配置 ======================
    EMPTY_CACHE_INTERVAL = 10  # 更频繁的GPU缓存清理
    USE_GRADIENT_CHECKPOINTING = False  # 避免维度问题

    # ====================== 保存和日志配置 ======================
    MODEL_SAVE_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    RESULT_DIR = "./results"
    SAVE_INTERVAL = 10
    LOG_INTERVAL = 20

    # ====================== 验证和可视化配置 ======================
    VAL_INTERVAL = 1
    VIS_INTERVAL = 10
    VIS_NUM_SAMPLES = 3

    # ====================== 评估配置 ======================
    EVALUATE_EDGES = False  # 是否评估边缘性能
    EDGE_THRESHOLD = 0.08  # 边缘检测阈值

    # ====================== 实验配置 ======================
    USE_WANDB = False
    EXPERIMENT_NAME = "lightweight_edge_net"

    def __init__(self):
        self._create_directories()
        self._validate_config()


    def _create_directories(self):
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

    def _validate_config(self):
        assert self.NUM_CLASSES > 0, "NUM_CLASSES must be positive"
        assert self.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert self.INPUT_SIZE > 0, "INPUT_SIZE must be positive"
        assert self.HIGH_FREQ_CHANNELS > 0, "HIGH_FREQ_CHANNELS must be positive"
        assert self.SEMANTIC_CHANNELS > 0, "SEMANTIC_CHANNELS must be positive"
        assert self.FUSION_CHANNELS >= max(self.HIGH_FREQ_CHANNELS, self.SEMANTIC_CHANNELS), \
            "FUSION_CHANNELS should be >= max(HIGH_FREQ_CHANNELS, SEMANTIC_CHANNELS)"

    def get_effective_batch_size(self):
        return self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS

    def get_input_shape(self):
        return (3, self.INPUT_SIZE, self.INPUT_SIZE)

    def get_memory_optimized_config(self):
        return {
            'batch_size': self.BATCH_SIZE,
            'gradient_accumulation_steps': self.GRADIENT_ACCUMULATION_STEPS,
            'mixed_precision': self.MIXED_PRECISION,
            'empty_cache_interval': self.EMPTY_CACHE_INTERVAL,
            'max_spatial_size': 32,
        }


    def get_stable_loss_config(self, level='stable'):
        """获取稳定损失配置"""
        if level == 'stable':
            return {
                'BOUNDARY_LOSS_WEIGHT': self.BOUNDARY_LOSS_WEIGHT_STABLE,
                'GRADIENT_LOSS_WEIGHT': self.GRADIENT_LOSS_WEIGHT_MINIMAL,
                'CONTRASTIVE_LOSS_WEIGHT': self.CONTRASTIVE_LOSS_WEIGHT_MINIMAL,
                'LOSS_TYPE': 'stable'
            }
        elif level == 'minimal':
            return {
                'BOUNDARY_LOSS_WEIGHT': self.BOUNDARY_LOSS_WEIGHT_MINIMAL,
                'GRADIENT_LOSS_WEIGHT': self.GRADIENT_LOSS_WEIGHT_ULTRA_MINIMAL,
                'CONTRASTIVE_LOSS_WEIGHT': self.CONTRASTIVE_LOSS_WEIGHT_MINIMAL,
                'LOSS_TYPE': 'stable'
            }
        elif level == 'ultra_minimal':
            return {
                'BOUNDARY_LOSS_WEIGHT': self.BOUNDARY_LOSS_WEIGHT_ULTRA_MINIMAL,
                'GRADIENT_LOSS_WEIGHT': self.GRADIENT_LOSS_WEIGHT_ULTRA_MINIMAL,
                'CONTRASTIVE_LOSS_WEIGHT': 0.05,
                'LOSS_TYPE': 'stable'
            }
        else:
            raise ValueError(f"Unknown stable loss level: {level}")

    def update_for_ablation(self, **kwargs):
        """为消融研究更新配置"""
        # 内存优化设置
        self.BATCH_SIZE = min(kwargs.get('BATCH_SIZE', self.BATCH_SIZE), 8)
        self.GRADIENT_ACCUMULATION_STEPS = max(8, self.GRADIENT_ACCUMULATION_STEPS)
        self.MIXED_PRECISION = True
        self.EMPTY_CACHE_INTERVAL = 5  # 更频繁的缓存清理
        self.NUM_WORKERS = 0
        self.PIN_MEMORY = False

        # 应用其他修改
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        print(f"消融研究配置更新: batch_size={self.BATCH_SIZE}, "
              f"effective_batch_size={self.get_effective_batch_size()}")
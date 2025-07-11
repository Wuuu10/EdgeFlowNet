# EdgeFlowNet: A Lightweight, Dynamic Edge-Aware Network for Precise Water Surface Boundary Detection

A PyTorch implementation of EdgeFlowNet, a specialized lightweight dual-branch network for water surface boundary detection with dynamic importance weighting and hybrid classical-deep learning integration.

## 🌟 Key Features

- **Ultra-Lightweight Design**: As low as 43.2K parameters while achieving 96.45% mIoU
- **Hybrid Architecture**: Classical edge operators (Sobel, Laplacian) integrated with deep learning
- **Dynamic Importance Weighting (DIW)**: Spatially-adaptive fusion of edge and semantic features
- **Environment-Robust Design**: Tested across 192 environmental conditions (day/night, weather variations)
- **Three Optimized Configurations**: Ultra, Efficient, and Accurate variants for different deployment needs
- **Real-Time Performance**: Up to 89.3 FPS on standard hardware

## 🏗️ Architecture Overview

EdgeFlowNet employs a dual-branch architecture that integrates classical edge detection with deep learning:

```
Input Water Surface Image (H×W×3)
    ↓
┌─────────────────────┬─────────────────────┐
│  High-Frequency     │   Semantic Branch   │
│  Branch             │                     │
│ • Classical Ops     │ • Encoder-Decoder   │
│   (Sobel, Laplacian)│ • Context Aggreg.   │
│ • Multi-Scale Edge  │ • Skip Connections  │
│ • Residual Blocks   │ • Efficient Upsampl.│
└─────────────────────┴─────────────────────┘
           ↓
    Dynamic Importance Weighting (DIW)
    • Edge-Aware Attention
    • Memory-Efficient Cross-Attention  
    • Spatially-Adaptive Fusion
           ↓
    Edge Enhancement
           ↓
   Segmentation Head
           ↓
    Binary Edge Map (H×W×1)
```

## ⚡ Model Variants

| Model | Parameters | Model Size | mIoU | FPS | Use Case |
|-------|------------|------------|------|-----|----------|
| **EdgeFlowNet-Ultra** | 43.2K | 0.21MB | 96.45% | 89.3 | Extreme Efficiency |
| **EdgeFlowNet-Efficient** | 74.0K | 0.39MB | 97.83% | 76.8 | Balanced Performance |
| **EdgeFlowNet-Accurate** | 118.7K | 0.53MB | 98.24% | 68.9 | Maximum Precision |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/lightweight-edge-water-segmentation.git
cd lightweight-edge-water-segmentation
pip install -r requirements.txt
```

### Dataset Preparation

EdgeFlowNet is designed for the **WaterScenes dataset**, which contains 54,120 high-resolution images across 192 environmental condition combinations:

1. Organize your dataset in the following structure:
```
WaterScenes/
├── image/                 # RGB water surface images
├── SegmentationClassPNG/  # Binary edge ground truth masks  
├── train.txt             # Training file list (70% split)
├── val.txt               # Validation file list (15% split)
└── test.txt              # Test file list (15% split)
```

2. Update data paths in `config.py`:
```python
DATA_ROOT = "/path/to/WaterScenes"
```

**Note**: The WaterScenes dataset covers diverse conditions including:
- **Temporal**: Day/Dusk/Night scenarios
- **Weather**: Sunny/Overcast/Rainy/Snowy conditions  
- **Lighting**: Various illumination and reflection conditions
- **Water Types**: Rivers, lakes, coastal areas with different surface characteristics

### Training

#### Train with paper configurations:
```bash
# Ultra-lightweight model (43.2K params, 96.45% mIoU)
python main.py --mode train --config ultra

# Efficient model (74.0K params, 97.83% mIoU)  
python main.py --mode train --config efficient

# High-accuracy model (118.7K params, 98.24% mIoU)
python main.py --mode train --config accurate
```

#### Training with paper settings:
```bash
# EdgeFlowNet-Ultra: batch_size=32, lr=1e-3
python train.py --config ultra --batch_size 32 --learning_rate 1e-3

# EdgeFlowNet-Efficient: batch_size=24, lr=8e-4
python train.py --config efficient --batch_size 24 --learning_rate 8e-4

# EdgeFlowNet-Accurate: batch_size=16, lr=5e-4  
python train.py --config accurate --batch_size 16 --learning_rate 5e-4
```

### Inference and Validation

```bash
# Validate a single model
python main.py --mode validate --checkpoint ./checkpoints/best_model.pth --visualize

# Compare multiple models
python main.py --mode compare --checkpoint ./checkpoints/ --visualize

# Performance benchmarking
python main.py --mode benchmark --checkpoint ./checkpoints/best_model.pth
```

## 🔬 Ablation Studies

The framework includes comprehensive ablation study tools:

```bash
# Quick ablation study
python main.py --mode ablation --ablation_mode quick

# Full ablation analysis
python main.py --mode ablation --ablation_mode full

# Specific component studies
python main.py --mode ablation --ablation_mode channel  # Channel configurations
python main.py --mode ablation --ablation_mode module   # Module ablations
python main.py --mode ablation --ablation_mode edge     # Edge processing
```

### Ablation Components

- **Channel Sensitivity**: Impact of different channel configurations
- **DIW Module Variants**: Standard, ultra-lightweight, simplified, minimal
- **Loss Function Analysis**: Boundary loss, gradient loss, contrastive loss
- **Edge Enhancement**: Different edge detection and enhancement strategies
- **Module Importance**: Progressive refinement, residual connections

## 📊 Configuration Details

### EdgeFlowNet-Ultra Configuration
```python
# Channel Configuration
HIGH_FREQ_CHANNELS = 16
SEMANTIC_CHANNELS = 20  
FUSION_CHANNELS = 24

# Architecture Settings
DIW_MODULE_TYPE = 'lightweight'
USE_PROGRESSIVE_REFINEMENT = False
EDGE_DETECTION_LAYERS = 1

# Training Settings  
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
BOUNDARY_LOSS_WEIGHT = 1.0
```

### EdgeFlowNet-Efficient Configuration
```python
# Channel Configuration
HIGH_FREQ_CHANNELS = 20
SEMANTIC_CHANNELS = 24
FUSION_CHANNELS = 32

# Architecture Settings
DIW_MODULE_TYPE = 'standard'
USE_PROGRESSIVE_REFINEMENT = False
EDGE_DETECTION_LAYERS = 2

# Training Settings
BATCH_SIZE = 24  
LEARNING_RATE = 8e-4
BOUNDARY_LOSS_WEIGHT = 1.0
```

### EdgeFlowNet-Accurate Configuration
```python
# Channel Configuration
HIGH_FREQ_CHANNELS = 24
SEMANTIC_CHANNELS = 32
FUSION_CHANNELS = 48

# Architecture Settings
DIW_MODULE_TYPE = 'enhanced'
USE_PROGRESSIVE_REFINEMENT = True
EDGE_DETECTION_LAYERS = 2

# Training Settings
BATCH_SIZE = 16
LEARNING_RATE = 5e-4  
BOUNDARY_LOSS_WEIGHT = 1.0
GRADIENT_LOSS_WEIGHT = 0.5
```

## 🎯 Performance Analysis

### Model Profiling
```bash
# Comprehensive model analysis
python main.py --mode profile --checkpoint ./checkpoints/best_model.pth --save_reports

# Memory and speed benchmarking
python validate.py --checkpoint ./checkpoints/best_model.pth --visualize
```

### Key Metrics
- **FLOPs and Parameters**: Computational complexity analysis
- **Inference Speed**: FPS measurements on different hardware
- **Memory Usage**: GPU memory consumption profiling
- **Efficiency Scores**: Comprehensive efficiency evaluation

## 📁 Project Structure

```
src/
├── main.py              # Main entry point
├── config.py            # Configuration management
├── train.py             # Training framework
├── validate.py          # Validation and benchmarking
├── ablation.py          # Ablation study framework
├── models/              # Model architectures
│   ├── fusion_net.py    # Main network architecture
│   ├── high_freq_branch.py  # High-frequency branch
│   ├── semantic_branch.py   # Semantic context branch
│   └── diw_module.py    # Dynamic importance weighting
├── data/                # Data loading and augmentation
│   └── custom_dataset.py
└── utils/               # Utility functions
    ├── loss.py          # Loss functions
    ├── metrics.py       # Evaluation metrics
    └── visualization.py # Results visualization
```

## 🔬 Technical Innovations

### 1. Dynamic Importance Weighting (DIW) Module
- **Spatially-adaptive fusion**: Learns to balance edge and semantic features based on local conditions
- **Components**: Edge-Aware Attention + Memory-Efficient Cross-Attention + Dynamic Weight Generation
- **Performance gain**: +1.68% mIoU improvement over static fusion with only +2.9K parameters

### 2. Hybrid Classical-Deep Learning Architecture  
- **Classical operators**: Integrated Sobel (X/Y) and Laplacian filters with learnable fusion
- **Performance**: +1.10% mIoU improvement over pure deep learning approach
- **Efficiency**: Leverages decades of signal processing knowledge instead of learning from scratch

### 3. Multi-Component Loss Function
```python
Total Loss = BCE + Boundary Loss + Gradient Loss + Focal Loss + Dice Loss
λ_boundary=1.0, λ_gradient=0.5, λ_focal=1.0, λ_dice=1.0
```
- **Boundary-sensitive loss**: +1.53% mIoU (largest single improvement)
- **Gradient consistency**: +0.55% mIoU for smooth edge preservation
- **Combined improvement**: +2.62% mIoU over baseline BCE loss

### 4. Binary Classification Optimization
- **Task-specific design**: Optimized for water-land boundary detection
- **Parameter efficiency**: 31.3× fewer parameters than general-purpose lightweight methods
- **Performance**: Competitive accuracy with orders of magnitude fewer parameters

## 📈 Experimental Results

### Performance Comparison on WaterScenes Dataset

| Method | Parameters | Model Size | mIoU | Edge F1 | Water F1 | FPS | Inference Time |
|--------|------------|------------|------|---------|----------|-----|----------------|
| **EdgeFlowNet-Ultra** | 43.2K | 0.21MB | 96.45% | 94.23% | 97.86% | 89.3 | 11.2ms |
| **EdgeFlowNet-Efficient** | 74.0K | 0.39MB | 97.83% | 95.67% | 98.91% | 76.8 | 13.0ms |
| **EdgeFlowNet-Accurate** | 118.7K | 0.53MB | 98.24% | 96.18% | 99.12% | 68.9 | 14.5ms |
| PiDiNet-v2 | 156K | 0.61MB | 97.16% | 95.42% | 97.58% | 42.1 | 23.7ms |
| LDNet | 95K | 0.42MB | 97.52% | 94.93% | 98.23% | 58.7 | 17.0ms |
| MobileNetV2-Seg | 3,504K | 16.1MB | 97.19% | 93.84% | 97.45% | 63.4 | 15.8ms |
| ESPNet | 470K | 1.8MB | 96.63% | 92.78% | 96.91% | 35.2 | 28.4ms |
| EDTER-Tiny | 890K | 3.4MB | 98.16% | 95.89% | 98.47% | 51.3 | 19.5ms |

### Key Performance Highlights

- **EdgeFlowNet-Efficient vs PiDiNet-v2**: 
  - 0.67% higher mIoU (97.83% vs 97.16%)
  - 52.6% fewer parameters (74K vs 156K)
  - 82% faster processing (76.8 FPS vs 42.1 FPS)

- **Parameter Efficiency**: 
  - EdgeFlowNet-Ultra achieves 96.45% mIoU with only 43.2K parameters
  - EdgeFlowNet-Accurate: 98.24% mIoU with 118.7K parameters (86.7% fewer than EDTER-Tiny)

- **Environmental Robustness**: 
  - Tested across 192 environmental condition combinations
  - Standard deviation: 0.8% across temporal conditions, 0.5% across lighting variations

### Ablation Study Results

#### Dual-Branch Architecture Analysis
| Configuration | Parameters | mIoU | Edge F1 | Water F1 | FPS |
|---------------|------------|------|---------|----------|-----|
| HF Branch Only | 31.2K | 94.87% | 91.23% | 96.15% | 112.3 |
| Semantic Branch Only | 35.6K | 95.92% | 89.76% | 97.34% | 98.7 |
| Dual-Branch w/o Classical | 68.4K | 96.73% | 93.45% | 97.89% | 82.1 |
| Dual-Branch w/o DIW | 71.2K | 96.15% | 92.34% | 97.56% | 78.5 |
| **Full Model** | **74.0K** | **97.83%** | **95.67%** | **98.91%** | **76.8** |

#### Classical Edge Operator Contribution
| Configuration | mIoU | Edge Precision | Edge Recall | Edge F1 | Params Δ |
|---------------|------|----------------|-------------|---------|-----------|
| No classical operators | 96.73% | 93.82% | 93.08% | 93.45% | 0 |
| Sobel only | 97.21% | 94.76% | 94.15% | 94.45% | +0.8K |
| Sobel + Laplacian (learnable) | 97.83% | 95.89% | 95.45% | 95.67% | +2.1K |

#### DIW Module Analysis  
| DIW Configuration | Parameters | mIoU | Edge F1 | FPS | mIoU Δ |
|-------------------|------------|------|---------|-----|---------|
| No Fusion (Concat) | 71.2K | 96.15% | 92.34% | 78.5 | 0 |
| Static Weighting | 71.8K | 96.78% | 93.12% | 77.9 | +0.63% |
| DIW w/o Cross-Attention | 72.9K | 97.34% | 94.58% | 77.2 | +1.19% |
| **Full DIW** | **74.0K** | **97.83%** | **95.67%** | **76.8** | **+1.68%** |

## 🛠️ Development

### Adding New Modules
1. Implement your module in `models/`
2. Update the configuration in `config.py`
3. Integrate with the fusion network in `fusion_net.py`
4. Add ablation studies in `ablation.py`

### Custom Datasets
1. Modify the dataset class in `data/custom_dataset.py`
2. Update data paths and class configurations
3. Adjust loss functions if needed

## 📋 Requirements

- Python 3.7+
- PyTorch 2.0+
- NVIDIA GPU with CUDA 10.2+ (for training)
- 4GB+ GPU memory (training), 2GB+ (inference)

## 🗂️ WaterScenes Dataset

EdgeFlowNet was evaluated on the comprehensive **WaterScenes dataset**:

- **Scale**: 54,120 high-resolution water surface images
- **Environmental Coverage**: 192 condition combinations
  - **Temporal**: Day, Dusk, Night scenarios
  - **Weather**: Sunny, Overcast, Rainy, Snowy conditions
  - **Lighting**: Various illumination and reflection states
  - **Water Types**: Rivers, lakes, coastal areas
- **Split**: Training (70%), Validation (15%), Test (15%)
- **Annotations**: Sub-pixel precision boundary labels for marine navigation
- **Advantages over general datasets**: 
  - ADE20K has 73% boundary annotation errors >5 pixels
  - WaterScenes provides water-level perspectives vs. distant elevated views
  - Includes critical night navigation and adverse weather scenarios

## 🔧 Advanced Usage

### Training Details (Paper Settings)
```bash
# Hardware: NVIDIA RTX 3090 GPU
# Framework: PyTorch 2.0 with AdamW optimizer
# Scheduler: Cosine annealing with 10-epoch warm-up

# Data Augmentation (following Zhang et al. 2024 best practices):
# - Random resized cropping (scale 0.5-2.0)  
# - Horizontal flipping (50% probability)
# - Color jittering (±30% brightness/contrast/saturation)
# - Rotation (±15°)
# - Gaussian blur for robustness
```

### Speed Benchmarking Protocol
```python
# Following standardized protocols from recent benchmarking studies
# - 100 warm-up iterations
# - 1000-run averages on consistent hardware
# - Multiple batch size testing for throughput analysis
```

### Memory Optimization
- **Mixed precision training (FP16)**: Enabled for all configurations
- **Gradient accumulation**: Adaptive based on available memory
- **Efficient data loading**: Optimized for WaterScenes dataset characteristics
- **Dynamic cache management**: Prevents OOM during long training sessions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Lightweight Edge-Aware Water Segmentation Network with Dynamic Importance Weighting},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2024}
}
```

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Inspired by recent advances in lightweight network design
- Built upon state-of-the-art segmentation methodologies

## 📞 Contact

For questions and feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).

---

**Keywords**: Lightweight Networks, Water Segmentation, Edge Detection, Real-time Inference, Mobile Deployment, DIW Module
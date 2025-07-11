import torch
import torch.nn as nn
import torch.nn.functional as F
from .high_freq_branch import HighFreqBranch
from .semantic_branch import SemanticBranch
from .diw_module import get_diw_module


class LightweightRefinementModule(nn.Module):

    def __init__(self, config):
        super(LightweightRefinementModule, self).__init__()

        channels = config.FUSION_CHANNELS

        self.multi_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(channels + channels // 4, channels, 1, bias=False)

        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        multi_feat = self.multi_scale(x)
        multi_feat = F.interpolate(multi_feat, size=(h, w), mode='bilinear', align_corners=False)

        edge_weight = self.edge_enhance(x)
        enhanced_x = x * (1.0 + edge_weight)

        combined = torch.cat([enhanced_x, multi_feat], dim=1)
        refined = self.fusion(combined)

        output = refined + x * self.residual_weight

        return output


class SegmentationHead(nn.Module):

    def __init__(self, config):
        super(SegmentationHead, self).__init__()

        in_channels = config.FUSION_CHANNELS
        num_classes = config.NUM_CLASSES

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)


class EdgeEnhancementModule(nn.Module):

    def __init__(self, config):
        super(EdgeEnhancementModule, self).__init__()

        channels = config.FUSION_CHANNELS

        self.edge_detector = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.edge_strength = config.EDGE_ENHANCEMENT_STRENGTH

    def forward(self, x):
        edge_map = self.edge_detector(x)
        enhanced = x * (1.0 + edge_map * self.edge_strength)
        return enhanced, edge_map


class ResidualConnection(nn.Module):

    def __init__(self, channels):
        super(ResidualConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)


class FusionNet(nn.Module):

    def __init__(self, config):
        super(FusionNet, self).__init__()

        self.config = config

        self.high_freq_branch = HighFreqBranch(config)
        self.semantic_branch = SemanticBranch(config)

        self.diw_module = get_diw_module(config)

        if self.diw_module is None:
            self.simple_fusion = nn.Conv2d(
                config.HIGH_FREQ_CHANNELS + config.SEMANTIC_CHANNELS,
                config.FUSION_CHANNELS,
                1, bias=False
            )

        self.edge_enhancer = EdgeEnhancementModule(config)

        use_progressive_refinement = getattr(config, 'USE_PROGRESSIVE_REFINEMENT', False)
        if use_progressive_refinement:
            self.refinement = LightweightRefinementModule(config)

        use_residual_connections = getattr(config, 'USE_RESIDUAL_CONNECTIONS', False)
        if use_residual_connections:
            self.residual_connection = ResidualConnection(config.FUSION_CHANNELS)

        self.seg_head = SegmentationHead(config)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _align_features(self, high_freq_feat, semantic_feat):
        if high_freq_feat.shape[2:] != semantic_feat.shape[2:]:
            target_h = min(high_freq_feat.shape[2], semantic_feat.shape[2])
            target_w = min(high_freq_feat.shape[3], semantic_feat.shape[3])
            target_size = (target_h, target_w)

            if high_freq_feat.shape[2:] != target_size:
                high_freq_feat = F.interpolate(
                    high_freq_feat, size=target_size,
                    mode='bilinear', align_corners=False
                )

            if semantic_feat.shape[2:] != target_size:
                semantic_feat = F.interpolate(
                    semantic_feat, size=target_size,
                    mode='bilinear', align_corners=False
                )

        return high_freq_feat, semantic_feat

    def forward(self, x, return_features=False):
        input_size = x.shape[2:]

        high_freq_feat = self.high_freq_branch(x)
        semantic_feat = self.semantic_branch(x)

        high_freq_feat, semantic_feat = self._align_features(high_freq_feat, semantic_feat)

        if self.diw_module is not None:
            hf_weight, sem_weight, hf_enhanced, sem_enhanced = self.diw_module(
                high_freq_feat, semantic_feat
            )

            fused_feat = hf_enhanced * hf_weight + sem_enhanced * sem_weight

        else:
            cat_feat = torch.cat([high_freq_feat, semantic_feat], dim=1)
            fused_feat = self.simple_fusion(cat_feat)
            hf_weight = sem_weight = None

        enhanced_feat, edge_map = self.edge_enhancer(fused_feat)

        if hasattr(self, 'refinement'):
            refined_feat = self.refinement(enhanced_feat)
        else:
            refined_feat = enhanced_feat

        if hasattr(self, 'residual_connection'):
            refined_feat = self.residual_connection(refined_feat)

        logits = self.seg_head(refined_feat)

        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size,
                mode='bilinear', align_corners=False
            )

        if return_features:
            features = {
                'high_freq_feat': high_freq_feat,
                'semantic_feat': semantic_feat,
                'fused_feat': refined_feat,
                'edge_map': edge_map
            }

            if hf_weight is not None:
                features['hf_weight'] = hf_weight
                features['sem_weight'] = sem_weight

            return logits, features

        return logits

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        hf_params = sum(p.numel() for p in self.high_freq_branch.parameters())
        sem_params = sum(p.numel() for p in self.semantic_branch.parameters())

        if self.diw_module is not None:
            diw_params = sum(p.numel() for p in self.diw_module.parameters())
        else:
            diw_params = sum(p.numel() for p in self.simple_fusion.parameters())

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'high_freq_params': hf_params,
            'semantic_params': sem_params,
            'diw_params': diw_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

    def get_feature_maps(self, x):
        with torch.no_grad():
            return self.forward(x, return_features=True)

    def get_complexity_info(self):
        info = self.get_model_info()

        architecture_info = {
            'use_diw_module': self.diw_module is not None,
            'use_simplified_diw': self.config.USE_SIMPLIFIED_DIW if self.diw_module is not None else False,
            'use_progressive_refinement': hasattr(self, 'refinement'),
            'use_residual_connections': hasattr(self, 'residual_connection'),
            'high_freq_channels': self.config.HIGH_FREQ_CHANNELS,
            'semantic_channels': self.config.SEMANTIC_CHANNELS,
            'fusion_channels': self.config.FUSION_CHANNELS
        }

        info.update(architecture_info)
        return info


class UltraLightweightFusionNet(nn.Module):

    def __init__(self, config):
        super(UltraLightweightFusionNet, self).__init__()

        config = config.__class__()
        config.HIGH_FREQ_CHANNELS = config.HIGH_FREQ_CHANNELS // 3
        config.SEMANTIC_CHANNELS = config.SEMANTIC_CHANNELS // 3
        config.FUSION_CHANNELS = config.FUSION_CHANNELS // 3

        config.USE_SIMPLIFIED_DIW = True
        config.USE_PROGRESSIVE_REFINEMENT = False

        self.high_freq_branch = HighFreqBranch(config)
        self.semantic_branch = SemanticBranch(config)
        self.diw_module = get_diw_module(config)
        self.seg_head = SegmentationHead(config)

    def forward(self, x):
        high_freq_feat = self.high_freq_branch(x)
        semantic_feat = self.semantic_branch(x)

        if high_freq_feat.shape[2:] != semantic_feat.shape[2:]:
            target_size = (
                min(high_freq_feat.shape[2], semantic_feat.shape[2]),
                min(high_freq_feat.shape[3], semantic_feat.shape[3])
            )
            high_freq_feat = F.interpolate(
                high_freq_feat, size=target_size,
                mode='bilinear', align_corners=False
            )
            semantic_feat = F.interpolate(
                semantic_feat, size=target_size,
                mode='bilinear', align_corners=False
            )

        if self.diw_module is not None:
            hf_weight, sem_weight, hf_enhanced, sem_enhanced = self.diw_module(
                high_freq_feat, semantic_feat
            )
            fused_feat = hf_enhanced * hf_weight + sem_enhanced * sem_weight
        else:
            fused_feat = high_freq_feat + semantic_feat

        logits = self.seg_head(fused_feat)

        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:],
                mode='bilinear', align_corners=False
            )

        return logits


def get_fusion_net(config, ultra_lightweight=False):
    if ultra_lightweight:
        return UltraLightweightFusionNet(config)
    else:
        return FusionNet(config)
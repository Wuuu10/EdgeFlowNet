import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(self.bn(x))
        return x


class LightweightChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(LightweightChannelAttention, self).__init__()
        reduced_channels = max(2, channels // reduction_ratio)
        print(f"LightweightChannelAttention: channels={channels}, reduced_channels={reduced_channels}")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.avg_pool(x)
        att = self.fc(att)
        return x * att


class EdgeAwareAttention(nn.Module):
    def __init__(self, channels):
        super(EdgeAwareAttention, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.channel_att = LightweightChannelAttention(channels)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        edge_map = self.edge_conv(x)
        x_ca = self.channel_att(x)
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_att(spatial_in)
        edge_enhanced = x_ca * (1.0 + edge_map * self.alpha)
        spatial_enhanced = edge_enhanced * spatial_att * self.beta
        return spatial_enhanced, edge_map


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, channels, attention_dim=8, max_spatial_size=32):
        super(MemoryEfficientCrossAttention, self).__init__()
        self.attention_dim = attention_dim
        self.scale = attention_dim ** -0.5
        self.max_spatial_size = max_spatial_size
        self.to_q = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.to_k = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _adaptive_downsample(self, x, target_size):
        if x.size(2) <= target_size and x.size(3) <= target_size:
            return x, 1
        scale_h = max(1, x.size(2) // target_size)
        scale_w = max(1, x.size(3) // target_size)
        scale_factor = max(scale_h, scale_w)
        new_h = x.size(2) // scale_factor
        new_w = x.size(3) // scale_factor
        x_down = F.adaptive_avg_pool2d(x, (new_h, new_w))
        return x_down, scale_factor

    def forward(self, x, y):
        B, C, H, W = x.shape
        x_down, scale_factor_x = self._adaptive_downsample(x, self.max_spatial_size)
        y_down, scale_factor_y = self._adaptive_downsample(y, self.max_spatial_size)
        H_d, W_d = x_down.shape[2:]
        q = self.to_q(x_down).flatten(2)
        k = self.to_k(y_down).flatten(2)
        v = self.to_v(y_down).flatten(2)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H_d, W_d)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        out = self.out_proj(out)
        return self.gamma * out + x


class UltraLightweightDIWModule(nn.Module):
    def __init__(self, config):
        super(UltraLightweightDIWModule, self).__init__()
        high_freq_ch = config.HIGH_FREQ_CHANNELS
        semantic_ch = config.SEMANTIC_CHANNELS
        fusion_ch = config.FUSION_CHANNELS
        self.high_freq_proj = nn.Conv2d(high_freq_ch, fusion_ch, 1, bias=False)
        self.semantic_proj = nn.Conv2d(semantic_ch, fusion_ch, 1, bias=False)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_ch * 2, fusion_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_ch // 4, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, high_freq_feat, semantic_feat):
        if high_freq_feat.shape[2:] != semantic_feat.shape[2:]:
            target_size = (
                min(high_freq_feat.shape[2], semantic_feat.shape[2]),
                min(high_freq_feat.shape[3], semantic_feat.shape[3])
            )
            high_freq_feat = F.interpolate(
                high_freq_feat, size=target_size, mode='bilinear', align_corners=False
            )
            semantic_feat = F.interpolate(
                semantic_feat, size=target_size, mode='bilinear', align_corners=False
            )
        hf_aligned = self.high_freq_proj(high_freq_feat)
        sem_aligned = self.semantic_proj(semantic_feat)
        hf_global = F.adaptive_avg_pool2d(hf_aligned, 1)
        sem_global = F.adaptive_avg_pool2d(sem_aligned, 1)
        global_concat = torch.cat([hf_global, sem_global], dim=1)
        weights = self.weight_generator(global_concat)
        hf_weight = weights[:, 0:1, :, :].expand_as(hf_aligned)
        sem_weight = weights[:, 1:2, :, :].expand_as(sem_aligned)
        return hf_weight, sem_weight, hf_aligned, sem_aligned


class DIWModule(nn.Module):
    def __init__(self, config):
        super(DIWModule, self).__init__()
        high_freq_ch = config.HIGH_FREQ_CHANNELS
        semantic_ch = config.SEMANTIC_CHANNELS
        fusion_ch = config.FUSION_CHANNELS
        reduction_ratio = config.DIW_REDUCTION_RATIO
        attention_dim = config.DIW_ATTENTION_DIM
        self.high_freq_proj = DepthwiseSeparableConv(high_freq_ch, fusion_ch)
        self.semantic_proj = DepthwiseSeparableConv(semantic_ch, fusion_ch)
        self.high_freq_enhance = EdgeAwareAttention(fusion_ch)
        self.semantic_enhance = EdgeAwareAttention(fusion_ch)
        self.cross_attention_hf = MemoryEfficientCrossAttention(fusion_ch, attention_dim)
        self.cross_attention_sem = MemoryEfficientCrossAttention(fusion_ch, attention_dim)
        self.weight_generator = nn.Sequential(
            nn.Conv2d(fusion_ch * 2, fusion_ch // reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(fusion_ch // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_ch // reduction_ratio, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.edge_weight_gen = nn.Sequential(
            nn.Conv2d(fusion_ch, fusion_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_ch // 4, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.edge_strength = config.EDGE_ENHANCEMENT_STRENGTH
        self.global_alpha = nn.Parameter(torch.ones(1))
        self.global_beta = nn.Parameter(torch.ones(1))

    def forward(self, high_freq_feat, semantic_feat):
        if high_freq_feat.shape[2:] != semantic_feat.shape[2:]:
            target_size = (
                min(high_freq_feat.shape[2], semantic_feat.shape[2]),
                min(high_freq_feat.shape[3], semantic_feat.shape[3])
            )
            high_freq_feat = F.interpolate(
                high_freq_feat, size=target_size, mode='bilinear', align_corners=False
            )
            semantic_feat = F.interpolate(
                semantic_feat, size=target_size, mode='bilinear', align_corners=False
            )
        hf_aligned = self.high_freq_proj(high_freq_feat)
        sem_aligned = self.semantic_proj(semantic_feat)
        hf_enhanced, hf_edge_map = self.high_freq_enhance(hf_aligned)
        sem_enhanced, sem_edge_map = self.semantic_enhance(sem_aligned)
        hf_cross = self.cross_attention_hf(hf_enhanced, sem_enhanced)
        sem_cross = self.cross_attention_sem(sem_enhanced, hf_enhanced)
        fused_features = torch.cat([hf_cross, sem_cross], dim=1)
        dynamic_weights = self.weight_generator(fused_features)
        hf_weight = dynamic_weights[:, 0:1, :, :]
        sem_weight = dynamic_weights[:, 1:2, :, :]
        edge_enhancement = self.edge_weight_gen(hf_cross)
        hf_weight_enhanced = hf_weight * (1.0 + edge_enhancement * self.edge_strength)
        total_weight = hf_weight_enhanced + sem_weight + 1e-8
        hf_weight_final = hf_weight_enhanced / total_weight
        sem_weight_final = sem_weight / total_weight
        hf_weight_final = hf_weight_final * self.global_alpha
        sem_weight_final = sem_weight_final * self.global_beta
        return hf_weight_final, sem_weight_final, hf_cross, sem_cross

    def get_attention_maps(self, high_freq_feat, semantic_feat):
        with torch.no_grad():
            hf_weight, sem_weight, _, _ = self.forward(high_freq_feat, semantic_feat)
            return {
                'high_freq_weight': hf_weight,
                'semantic_weight': sem_weight,
                'combined_weight': torch.cat([hf_weight, sem_weight], dim=1)
            }


class SimplifiedDIWModule(nn.Module):
    def __init__(self, config):
        super(SimplifiedDIWModule, self).__init__()
        high_freq_ch = config.HIGH_FREQ_CHANNELS
        semantic_ch = config.SEMANTIC_CHANNELS
        fusion_ch = config.FUSION_CHANNELS
        self.high_freq_proj = DepthwiseSeparableConv(high_freq_ch, fusion_ch, 1, 1, 0)
        self.semantic_proj = DepthwiseSeparableConv(semantic_ch, fusion_ch, 1, 1, 0)
        self.hf_channel_att = LightweightChannelAttention(fusion_ch, reduction_ratio=8)
        self.sem_channel_att = LightweightChannelAttention(fusion_ch, reduction_ratio=8)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_ch * 2, fusion_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_ch // 4, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, high_freq_feat, semantic_feat):
        if high_freq_feat.shape[2:] != semantic_feat.shape[2:]:
            target_size = (
                min(high_freq_feat.shape[2], semantic_feat.shape[2]),
                min(high_freq_feat.shape[3], semantic_feat.shape[3])
            )
            high_freq_feat = F.interpolate(
                high_freq_feat, size=target_size, mode='bilinear', align_corners=False
            )
            semantic_feat = F.interpolate(
                semantic_feat, size=target_size, mode='bilinear', align_corners=False
            )
        hf_aligned = self.high_freq_proj(high_freq_feat)
        sem_aligned = self.semantic_proj(semantic_feat)
        hf_enhanced = self.hf_channel_att(hf_aligned)
        sem_enhanced = self.sem_channel_att(sem_aligned)
        hf_global = F.adaptive_avg_pool2d(hf_enhanced, 1)
        sem_global = F.adaptive_avg_pool2d(sem_enhanced, 1)
        global_concat = torch.cat([hf_global, sem_global], dim=1)
        weights = self.weight_generator(global_concat)
        hf_weight = weights[:, 0:1, :, :].expand_as(hf_enhanced)
        sem_weight = weights[:, 1:2, :, :].expand_as(sem_enhanced)
        hf_final = hf_enhanced + hf_aligned * self.residual_weight
        sem_final = sem_enhanced + sem_aligned * self.residual_weight
        return hf_weight, sem_weight, hf_final, sem_final


def get_diw_module(config):
    if not config.USE_DIW_MODULE:
        return None
    use_simplified_diw = getattr(config, 'USE_SIMPLIFIED_DIW', False)
    diw_module_type = getattr(config, 'DIW_MODULE_TYPE', 'standard')
    if diw_module_type == 'ultra_lightweight':
        print("Ultra-Lightweight DIW")
        return UltraLightweightDIWModule(config)
    elif diw_module_type == 'simplified':
        print("Simplified DIW")
        return SimplifiedDIWModule(config)
    elif diw_module_type == 'minimal':
        print("Minimal DIW")
        return UltraLightweightDIWModule(config)
    elif use_simplified_diw:
        print("Simplified DIW")
        return SimplifiedDIWModule(config)
    else:
        print("Full DIW")
        return DIWModule(config)

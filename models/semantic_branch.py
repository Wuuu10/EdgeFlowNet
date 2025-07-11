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
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class LightweightPyramidPooling(nn.Module):

    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 4]):
        super(LightweightPyramidPooling, self).__init__()

        self.pool_sizes = pool_sizes
        branch_channels = out_channels // (len(pool_sizes) + 1)  # +1 for original features

        self.branches = nn.ModuleList()
        for pool_size in pool_sizes:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            ))

        self.original_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        total_channels = branch_channels * (len(pool_sizes) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        pyramid_feats = []

        for branch in self.branches:
            feat = branch(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_feats.append(feat)

        original_feat = self.original_branch(x)
        pyramid_feats.append(original_feat)

        concatenated = torch.cat(pyramid_feats, dim=1)
        output = self.fusion(concatenated)

        return output


class ContextAggregationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextAggregationModule, self).__init__()

        self.local_context = DepthwiseSeparableConv(in_channels, out_channels // 2, 3, 1, 1)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.out_channels = out_channels
        self.half_channels = out_channels // 2

        self.aggregation = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),  # 固定输出通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        local_feat = self.local_context(x)  # [B, out_channels//2, H, W]

        global_weight = self.global_context(x)
        global_enhanced = local_feat * global_weight

        avg_pool = torch.mean(global_enhanced, dim=1, keepdim=True)
        max_pool, _ = torch.max(global_enhanced, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)

        spatial_enhanced = global_enhanced * spatial_weight

        combined = torch.cat([local_feat, spatial_enhanced], dim=1)  # [B, out_channels, H, W]

        if combined.shape[1] != self.out_channels:
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = nn.Conv2d(
                    combined.shape[1], self.out_channels, 1, bias=False
                ).to(combined.device)
            combined = self.channel_adapter(combined)

        output = self.aggregation(combined)
        return output


class EfficientUpsampling(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(EfficientUpsampling, self).__init__()
        self.scale_factor = scale_factor

        self.upsample_conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=scale_factor * 2, stride=scale_factor,
            padding=scale_factor // 2, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.refine_conv = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = self.relu(self.bn(x))
        x = self.refine_conv(x)
        return x


class SemanticBranch(nn.Module):
    def __init__(self, config):
        super(SemanticBranch, self).__init__()

        self.config = config
        in_channels = 3
        base_channels = config.SEMANTIC_CHANNELS // 2
        out_channels = config.SEMANTIC_CHANNELS

        assert base_channels > 0, f"base_channels must be positive, got {base_channels}"
        assert out_channels >= base_channels, f"out_channels ({out_channels}) should >= base_channels ({base_channels})"
        print(f"base_channels={base_channels}, out_channels={out_channels}")

        self.encoder_stage1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1)
        )

        self.downsample1 = DepthwiseSeparableConv(base_channels, base_channels, 3, 2, 1)

        self.encoder_stage2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, 3, 1, 1),
            ContextAggregationModule(base_channels * 2, base_channels * 2)
        )

        self.downsample2 = DepthwiseSeparableConv(base_channels * 2, base_channels * 2, 3, 2, 1)

        self.bottleneck = nn.Sequential(
            LightweightPyramidPooling(base_channels * 2, base_channels * 2),
            ContextAggregationModule(base_channels * 2, base_channels * 2)
        )

        self.upsample1 = EfficientUpsampling(base_channels * 2, base_channels, 2)

        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 1, bias=False),  # 2*base + base
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.decoder_stage1 = ContextAggregationModule(base_channels, base_channels)

        self.upsample2 = EfficientUpsampling(base_channels, base_channels, 2)

        self.skip_fusion2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1, bias=False),  # base + base
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.decoder_stage2 = ContextAggregationModule(base_channels, base_channels)

        self.output_conv = nn.Sequential(
            DepthwiseSeparableConv(base_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.semantic_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.shape[2:]

        enc1 = self.encoder_stage1(x)
        down1 = self.downsample1(enc1)
        enc2 = self.encoder_stage2(down1)
        down2 = self.downsample2(enc2)
        bottleneck = self.bottleneck(down2)
        up1 = self.upsample1(bottleneck)

        if up1.shape[2:] != enc2.shape[2:]:
            up1 = F.interpolate(up1, size=enc2.shape[2:], mode='bilinear', align_corners=False)

        skip1 = torch.cat([up1, enc2], dim=1)
        fused1 = self.skip_fusion1(skip1)
        dec1 = self.decoder_stage1(fused1)

        up2 = self.upsample2(dec1)

        if up2.shape[2:] != enc1.shape[2:]:
            up2 = F.interpolate(up2, size=enc1.shape[2:], mode='bilinear', align_corners=False)

        skip2 = torch.cat([up2, enc1], dim=1)
        fused2 = self.skip_fusion2(skip2)
        dec2 = self.decoder_stage2(fused2)

        output = self.output_conv(dec2)

        semantic_weight = self.semantic_enhance(output)
        enhanced_output = output * semantic_weight

        if enhanced_output.shape[2:] != input_size:
            enhanced_output = F.interpolate(
                enhanced_output, size=input_size,
                mode='bilinear', align_corners=False
            )

        return enhanced_output

    def get_multi_scale_features(self, x):
        with torch.no_grad():
            enc1 = self.encoder_stage1(x)
            down1 = self.downsample1(enc1)
            enc2 = self.encoder_stage2(down1)
            down2 = self.downsample2(enc2)
            bottleneck = self.bottleneck(down2)

            return {
                'encoder_stage1': enc1,
                'encoder_stage2': enc2,
                'bottleneck': bottleneck,
                'multi_scale_sizes': [enc1.shape[2:], enc2.shape[2:], bottleneck.shape[2:]]
            }


class LightweightSemanticBranch(nn.Module):

    def __init__(self, config):
        super(LightweightSemanticBranch, self).__init__()

        in_channels = 3
        out_channels = config.SEMANTIC_CHANNELS
        mid_channels = out_channels // 2

        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels),
            DepthwiseSeparableConv(mid_channels, mid_channels, stride=2),
            DepthwiseSeparableConv(mid_channels, out_channels, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, mid_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        if decoded.shape[2:] != input_size:
            decoded = F.interpolate(
                decoded, size=input_size,
                mode='bilinear', align_corners=False
            )

        return decoded

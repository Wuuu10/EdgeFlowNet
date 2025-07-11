import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class EdgeDetectionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EdgeDetectionModule, self).__init__()

        self.main_conv = DepthwiseSeparableConv(in_channels, out_channels // 2)

        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        laplacian = torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(3, out_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        fusion_in_channels = out_channels // 2 + out_channels // 4
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        attention_channels = max(4, out_channels // 8)
        self.edge_attention = nn.Sequential(
            nn.Conv2d(out_channels, attention_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

        print(f"EdgeDetectionModule: out_channels={out_channels}, attention_channels={attention_channels}")

    def forward(self, x):
        main_feat = self.main_conv(x)

        x_gray = torch.mean(x, dim=1, keepdim=True)

        edge_x = F.conv2d(F.pad(x_gray, (1, 1, 1, 1)), self.sobel_x)
        edge_y = F.conv2d(F.pad(x_gray, (1, 1, 1, 1)), self.sobel_y)
        edge_lap = F.conv2d(F.pad(x_gray, (1, 1, 1, 1)), self.laplacian)

        edge_combined = torch.cat([edge_x, edge_y, edge_lap], dim=1)
        edge_feat = self.edge_conv(edge_combined)

        combined = torch.cat([main_feat, edge_feat], dim=1)
        fused = self.fusion(combined)

        attention = self.edge_attention(fused)
        output = fused * attention

        return output


class LightweightResidualBlock(nn.Module):

    def __init__(self, channels, reduction_ratio=4):
        super(LightweightResidualBlock, self).__init__()

        reduced_channels = max(4, channels // reduction_ratio)

        print(f"LightweightResidualBlock: channels={channels}, reduced_channels={reduced_channels}")

        self.conv1 = DepthwiseSeparableConv(channels, reduced_channels, 1, 1, 0)
        self.conv2 = DepthwiseSeparableConv(reduced_channels, reduced_channels, 3, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        se_channels = max(2, channels // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, se_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # SE注意力
        se_weight = self.se(out)
        out = out * se_weight

        out += residual
        out = self.relu(out)

        return out


class MultiScaleEdgeModule(nn.Module):

    def __init__(self, in_channels, out_channels, scales=[3, 5, 7]):
        super(MultiScaleEdgeModule, self).__init__()

        self.scales = scales
        scale_out_channels = out_channels // len(scales)

        self.scale_convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels, scale_out_channels,
                kernel_size=k, padding=k // 2
            ) for k in scales
        ])

        total_channels = scale_out_channels * len(scales)
        self.aggregation = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(out_channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale_feats = []
        for conv in self.scale_convs:
            scale_feats.append(conv(x))

        combined = torch.cat(scale_feats, dim=1)
        aggregated = self.aggregation(combined)

        edge_weight = self.edge_enhance(aggregated)
        enhanced = aggregated * (1.0 + edge_weight)

        return enhanced


class HighFreqBranch(nn.Module):

    def __init__(self, config):
        super(HighFreqBranch, self).__init__()

        self.config = config
        in_channels = 3
        base_channels = config.HIGH_FREQ_CHANNELS // 2
        out_channels = config.HIGH_FREQ_CHANNELS

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.edge_detection = EdgeDetectionModule(base_channels, base_channels)

        self.multi_scale_edge = MultiScaleEdgeModule(
            base_channels, base_channels,
            scales=[3, 5] if config.EDGE_DETECTION_LAYERS <= 2 else [3, 5, 7]
        )

        self.residual_blocks = nn.ModuleList([
            LightweightResidualBlock(base_channels)
            for _ in range(config.EDGE_DETECTION_LAYERS)
        ])

        self.feature_fusion = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, out_channels),
            LightweightResidualBlock(out_channels)
        )

        self.boundary_processor = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.output_conv = DepthwiseSeparableConv(out_channels, out_channels)

        self.global_edge_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.shape[2:]

        feat = self.init_conv(x)
        edge_feat = self.edge_detection(feat)
        multi_scale_feat = self.multi_scale_edge(feat)

        residual_feat = multi_scale_feat
        for block in self.residual_blocks:
            residual_feat = block(residual_feat)

        combined_feat = torch.cat([edge_feat, residual_feat], dim=1)
        fused_feat = self.feature_fusion(combined_feat)

        boundary_feat = self.boundary_processor(fused_feat)

        output_feat = self.output_conv(boundary_feat)

        global_weight = self.global_edge_enhance(output_feat)
        enhanced_output = output_feat * global_weight

        if enhanced_output.shape[2:] != input_size:
            enhanced_output = F.interpolate(
                enhanced_output, size=input_size,
                mode='bilinear', align_corners=False
            )

        return enhanced_output

    def get_edge_features(self, x):
        with torch.no_grad():
            feat = self.init_conv(x)
            edge_feat = self.edge_detection(feat)
            multi_scale_feat = self.multi_scale_edge(feat)

            return {
                'initial_features': feat,
                'edge_features': edge_feat,
                'multi_scale_features': multi_scale_feat
            }

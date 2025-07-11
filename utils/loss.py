import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if num_classes == 1:
            inputs = torch.sigmoid(inputs)
            targets = targets.float()
        else:
            inputs = F.softmax(inputs, dim=1)
            targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        inputs = inputs.reshape(inputs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1 - dice.mean()


class AdaptiveBoundaryLoss(nn.Module):
    def __init__(self, weight=3.0, edge_threshold=0.1, use_focal=True):
        super(AdaptiveBoundaryLoss, self).__init__()
        self.weight = weight
        self.edge_threshold = edge_threshold
        self.use_focal = use_focal

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        if self.use_focal:
            self.focal_loss = FocalLoss()

        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _get_boundary_mask(self, targets):
        targets_float = targets.float().unsqueeze(1)

        sobel_x = self.sobel_x.to(targets_float.device).to(targets_float.dtype)
        sobel_y = self.sobel_y.to(targets_float.device).to(targets_float.dtype)

        edge_x = F.conv2d(F.pad(targets_float, (1, 1, 1, 1)), sobel_x)
        edge_y = F.conv2d(F.pad(targets_float, (1, 1, 1, 1)), sobel_y)

        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        boundary_mask = (edge_magnitude > self.edge_threshold).float().squeeze(1)

        return boundary_mask

    def forward(self, inputs, targets, edge_weights=None):
        total_loss = 0.0

        if self.use_focal:
            focal_loss = self.focal_loss(inputs, targets)
            total_loss += focal_loss
        else:
            ce_loss = self.ce_loss(inputs, targets)
            total_loss += ce_loss

        dice_loss = self.dice_loss(inputs, targets)
        total_loss += dice_loss

        if self.weight > 0:
            boundary_mask = self._get_boundary_mask(targets)

            if edge_weights is not None:
                boundary_mask = boundary_mask * edge_weights

            if boundary_mask.sum() > 0:
                boundary_inputs = inputs.permute(0, 2, 3, 1)[boundary_mask > 0]
                boundary_targets = targets[boundary_mask > 0]

                if boundary_inputs.size(0) > 0:
                    boundary_ce = F.cross_entropy(boundary_inputs, boundary_targets)

                    boundary_ratio = boundary_mask.sum() / boundary_mask.numel()
                    adaptive_weight = self.weight * torch.clamp(boundary_ratio * 10, 0.5, 2.0)

                    total_loss += adaptive_weight * boundary_ce

        return total_loss


class BoundaryLoss(nn.Module):
    def __init__(self, weight=3.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _get_boundary_mask(self, targets):
        targets_float = targets.float().unsqueeze(1)

        sobel_x = self.sobel_x.to(targets_float.device).to(targets_float.dtype)
        sobel_y = self.sobel_y.to(targets_float.device).to(targets_float.dtype)

        edge_x = F.conv2d(F.pad(targets_float, (1, 1, 1, 1)), sobel_x)
        edge_y = F.conv2d(F.pad(targets_float, (1, 1, 1, 1)), sobel_y)

        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        boundary_mask = (edge_magnitude > 0.1).float().squeeze(1)

        return boundary_mask

    def forward(self, inputs, targets, edge_weights=None):
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)

        boundary_mask = self._get_boundary_mask(targets)

        if edge_weights is not None:
            boundary_mask = boundary_mask * edge_weights

        if boundary_mask.sum() > 0:
            boundary_mask_expanded = boundary_mask.unsqueeze(1).expand_as(inputs)

            boundary_inputs = inputs * boundary_mask_expanded
            boundary_targets_one_hot = F.one_hot(targets, inputs.size(1)).permute(0, 3, 1, 2).float()
            boundary_targets_masked = boundary_targets_one_hot * boundary_mask_expanded

            boundary_bce = F.binary_cross_entropy_with_logits(
                boundary_inputs, boundary_targets_masked, reduction='mean'
            )

            boundary_inputs_sigmoid = torch.sigmoid(boundary_inputs)
            intersection = torch.sum(boundary_inputs_sigmoid * boundary_targets_masked, dim=(0, 2, 3))
            union = torch.sum(boundary_inputs_sigmoid, dim=(0, 2, 3)) + torch.sum(boundary_targets_masked,
                                                                                  dim=(0, 2, 3))
            boundary_dice = 1 - (2.0 * intersection + 1.0) / (union + 1.0)
            boundary_dice = boundary_dice.mean()

            boundary_loss = boundary_bce + boundary_dice
        else:
            boundary_loss = 0.0

        total_loss = ce_loss + dice_loss + self.weight * boundary_loss
        return total_loss


class GradientConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(GradientConsistencyLoss, self).__init__()
        self.weight = weight

        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradients(self, x):
        if x.dim() == 4 and x.size(1) > 1:
            x = F.softmax(x, dim=1)[:, 1:2, :, :]
        elif x.dim() == 3:
            x = x.unsqueeze(1).float()

        sobel_x = self.sobel_x.to(x.device).to(x.dtype)
        sobel_y = self.sobel_y.to(x.device).to(x.dtype)

        grad_x = F.conv2d(F.pad(x, (1, 1, 1, 1)), sobel_x)
        grad_y = F.conv2d(F.pad(x, (1, 1, 1, 1)), sobel_y)

        return grad_x, grad_y

    def forward(self, inputs, targets):
        pred_grad_x, pred_grad_y = self._compute_gradients(inputs)
        target_grad_x, target_grad_y = self._compute_gradients(targets)

        # 梯度差异损失
        grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        total_grad_loss = grad_loss_x + grad_loss_y
        return self.weight * total_grad_loss


class EdgeAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=2.0, edge_temp=1.0):
        super(EdgeAwareContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.edge_temp = edge_temp
        self.projections = nn.ModuleDict()

    def _get_projection_key(self, in_channels, out_channels):
        return f"{in_channels}_{out_channels}"

    def _get_or_create_projection(self, in_channels, out_channels, device):
        key = self._get_projection_key(in_channels, out_channels)

        if key not in self.projections:
            self.projections[key] = nn.Linear(in_channels, out_channels).to(device)

        return self.projections[key]

    def forward(self, high_freq_feat, semantic_feat, targets=None):
        if high_freq_feat is None or semantic_feat is None:
            return torch.tensor(0.0,
                                device=high_freq_feat.device if high_freq_feat is not None else semantic_feat.device)

        try:
            hf_pool = F.adaptive_avg_pool2d(high_freq_feat, 1).flatten(1)  # [B, C_hf]
            sem_pool = F.adaptive_avg_pool2d(semantic_feat, 1).flatten(1)  # [B, C_sem]

            hf_channels = hf_pool.size(1)
            sem_channels = sem_pool.size(1)

            if hf_channels == sem_channels:
                hf_norm = F.normalize(hf_pool, p=2, dim=1)
                sem_norm = F.normalize(sem_pool, p=2, dim=1)
            else:
                target_dim = min(hf_channels, sem_channels)

                hf_proj = self._get_or_create_projection(hf_channels, target_dim, high_freq_feat.device)
                sem_proj = self._get_or_create_projection(sem_channels, target_dim, semantic_feat.device)

                hf_projected = hf_proj(hf_pool)
                sem_projected = sem_proj(sem_pool)

                hf_norm = F.normalize(hf_projected, p=2, dim=1)
                sem_norm = F.normalize(sem_projected, p=2, dim=1)

            similarity = torch.mm(hf_norm, sem_norm.t()) / self.temperature

            batch_size = similarity.size(0)
            targets_sim = torch.eye(batch_size, device=similarity.device)

            loss = F.binary_cross_entropy_with_logits(similarity, targets_sim)
            return loss

        except Exception as e:
            return torch.tensor(0.0, device=high_freq_feat.device, requires_grad=True)


class LightweightCombinedLoss(nn.Module):
    def __init__(self, config):
        super(LightweightCombinedLoss, self).__init__()

        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss()

        if config.USE_BOUNDARY_LOSS:
            boundary_weight = getattr(config, 'BOUNDARY_LOSS_WEIGHT', 3.0)

            if hasattr(config, 'USE_ADAPTIVE_BOUNDARY') and config.USE_ADAPTIVE_BOUNDARY:
                self.boundary_loss = AdaptiveBoundaryLoss(weight=boundary_weight)
            else:
                self.boundary_loss = BoundaryLoss(weight=boundary_weight)

        if config.USE_GRADIENT_LOSS:
            gradient_weight = getattr(config, 'GRADIENT_LOSS_WEIGHT', 1.0)
            self.gradient_loss = GradientConsistencyLoss(weight=gradient_weight)

        self.contrastive_loss = EdgeAwareContrastiveLoss()

        self.contrastive_weight = getattr(config, 'CONTRASTIVE_LOSS_WEIGHT', 0.3)
        self.edge_focal_weight = getattr(config, 'EDGE_FOCAL_LOSS_WEIGHT', 0.5)

        self.loss_history = {
            'ce_loss': [],
            'focal_loss': [],
            'dice_loss': [],
            'boundary_loss': [],
            'gradient_loss': [],
            'contrastive_loss': []
        }

    def forward(self, outputs, targets, high_freq_feat=None, semantic_feat=None, edge_weights=None):
        ce_loss = self.ce_loss(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        dice_loss = self.dice_loss(outputs, targets)

        ce_loss = torch.clamp(ce_loss, min=0.0)
        focal_loss = torch.clamp(focal_loss, min=0.0)
        dice_loss = torch.clamp(dice_loss, min=0.0)
        total_loss = ce_loss + focal_loss + dice_loss

        self.loss_history['ce_loss'].append(ce_loss.item())
        self.loss_history['focal_loss'].append(focal_loss.item())
        self.loss_history['dice_loss'].append(dice_loss.item())

        if hasattr(self, 'boundary_loss'):
            boundary_loss = self.boundary_loss(outputs, targets, edge_weights)
            boundary_loss = torch.clamp(boundary_loss, min=0.0)
            total_loss += boundary_loss
            self.loss_history['boundary_loss'].append(boundary_loss.item())

        if hasattr(self, 'gradient_loss'):
            gradient_loss = self.gradient_loss(outputs, targets)
            gradient_loss = torch.clamp(gradient_loss, min=0.0)
            total_loss += gradient_loss
            self.loss_history['gradient_loss'].append(gradient_loss.item())

        if high_freq_feat is not None and semantic_feat is not None:
            contrastive_loss = self.contrastive_loss(high_freq_feat, semantic_feat, targets)

            if contrastive_loss < 0:
                contrastive_loss = torch.tensor(0.0, device=contrastive_loss.device, requires_grad=True)
            else:
                contrastive_loss = torch.clamp(contrastive_loss, max=2.0)

            total_loss += self.contrastive_weight * contrastive_loss
            self.loss_history['contrastive_loss'].append(contrastive_loss.item())

        if edge_weights is not None:
            edge_mask = edge_weights > 1.5
            if edge_mask.sum() > 0:
                edge_outputs = outputs[edge_mask.unsqueeze(1).expand_as(outputs)].view(-1, outputs.size(1))
                edge_targets = targets[edge_mask].view(-1)

                if edge_outputs.size(0) > 0:
                    edge_focal_loss = self.focal_loss(edge_outputs, edge_targets)
                    edge_focal_loss = torch.clamp(edge_focal_loss, min=0.0)
                    total_loss += self.edge_focal_weight * edge_focal_loss

        total_loss = torch.clamp(total_loss, min=0.0)

        return total_loss

    def get_loss_summary(self):
        summary = {}
        for key, values in self.loss_history.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values[-100:]),
                    'std': np.std(values[-100:]),
                    'min': np.min(values[-100:]),
                    'max': np.max(values[-100:])
                }
        return summary


class StableLoss(nn.Module):
    def __init__(self, config):
        super(StableLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        if getattr(config, 'USE_BOUNDARY_LOSS', False):
            boundary_weight = min(getattr(config, 'BOUNDARY_LOSS_WEIGHT', 1.0), 1.5)
            self.boundary_loss = BoundaryLoss(weight=boundary_weight)
            self.use_boundary = True
        else:
            self.use_boundary = False

    def forward(self, outputs, targets, **kwargs):
        ce_loss = self.ce_loss(outputs, targets)
        dice_loss = self.dice_loss(outputs, targets)
        ce_loss = torch.clamp(ce_loss, min=0.0)
        dice_loss = torch.clamp(dice_loss, min=0.0)

        total_loss = ce_loss + dice_loss

        if self.use_boundary:
            boundary_loss = self.boundary_loss(outputs, targets, kwargs.get('edge_weights', None))
            boundary_loss = torch.clamp(boundary_loss, min=0.0, max=5.0)
            total_loss += boundary_loss

        return torch.clamp(total_loss, min=0.0)


def get_loss_function(config):
    loss_type = getattr(config, 'LOSS_TYPE', 'combined')
    if loss_type == 'stable':
        return StableLoss(config)
    elif loss_type == 'combined':
        return LightweightCombinedLoss(config)
    else:
        return StableLoss(config)
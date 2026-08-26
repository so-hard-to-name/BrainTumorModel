import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits, target, has_mask, eps=1e-6):
    """Soft dice loss, computed only over samples where has_mask is True."""
    if has_mask.sum() == 0:
        return logits.sum() * 0.0

    probs = torch.sigmoid(logits)
    probs = probs[has_mask]
    target = target[has_mask]

    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def masked_bce_loss(logits, target, has_mask):
    if has_mask.sum() == 0:
        return logits.sum() * 0.0
    logits = logits[has_mask]
    target = target[has_mask]
    return F.binary_cross_entropy_with_logits(logits, target)


class MultiTaskLoss(nn.Module):
    def __init__(self, cls_weight=1.0, seg_bce_weight=1.0, seg_dice_weight=1.0,
                 class_weights=None):   # weights can be adjusted
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, cls_logits, seg_logits, labels, masks, has_mask):
        cls_loss = F.cross_entropy(cls_logits, labels, weight=self.class_weights)

        bce = masked_bce_loss(seg_logits, masks, has_mask)
        dice = dice_loss(seg_logits, masks, has_mask)
        seg_loss = self.seg_bce_weight * bce + self.seg_dice_weight * dice

        total = self.cls_weight * cls_loss + seg_loss
        return {
            "total": total,
            "cls_loss": cls_loss.detach(),
            "seg_bce": bce.detach(),
            "seg_dice": dice.detach(),
        }
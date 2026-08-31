import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.brisc.dataset import Brisc2025Dataset
from src.brisc.models.multitask_model import MultiTaskModel

CLASS_NAMES = ["no_tumor", "glioma", "meningioma", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_DIRECTIONS = 3


def load_model(checkpoint_path, device, image_size, base_ch, embed_dim, block_num, num_heads):
    max_tokens = (image_size // 16) ** 2
    model = MultiTaskModel(
        in_channels=1, num_classes=NUM_CLASSES, num_seg_ch=1,
        base_ch=base_ch, embed_dim=embed_dim, block_num=block_num,
        num_heads=num_heads, max_tokens=max_tokens, num_directions=NUM_DIRECTIONS,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, max_qualitative=8):
    all_labels, all_preds = [], []
    dice_scores_overall = []
    dice_scores_per_class = {c: [] for c in range(NUM_CLASSES)}
    qualitative = []

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        has_mask = batch["has_mask"].to(device)
        direction = batch["direction"].to(device)
        label = batch["tumor_type"].to(device)

        cls_logits, seg_logits = model(image, direction=direction)
        pred_label = cls_logits.argmax(dim=1)

        all_labels.extend(label.cpu().tolist())
        all_preds.extend(pred_label.cpu().tolist())

        probs = torch.sigmoid(seg_logits)
        preds = (probs > 0.5).float()

        if has_mask.any():
            hm_preds = preds[has_mask]
            hm_gt = mask[has_mask]
            hm_labels = label[has_mask]

            inter = (hm_preds * hm_gt).sum(dim=(1, 2, 3))
            union = hm_preds.sum(dim=(1, 2, 3)) + hm_gt.sum(dim=(1, 2, 3))
            dice = (2 * inter + 1e-6) / (union + 1e-6)

            dice_scores_overall.extend(dice.cpu().tolist())
            for d, lbl in zip(dice.cpu().tolist(), hm_labels.cpu().tolist()):
                dice_scores_per_class[lbl].append(d)

        if len(qualitative) < max_qualitative:
            room = max_qualitative - len(qualitative)
            for i in range(min(image.size(0), room)):
                qualitative.append({
                    "image": image[i, 0].cpu().numpy(),
                    "gt_mask": mask[i, 0].cpu().numpy(),
                    "pred_mask": preds[i, 0].cpu().numpy(),
                    "has_mask": bool(has_mask[i].item()),
                    "label": CLASS_NAMES[label[i].item()],
                    "pred_label": CLASS_NAMES[pred_label[i].item()],
                })

    return all_labels, all_preds, dice_scores_overall, dice_scores_per_class, qualitative


def save_qualitative(qualitative, out_dir):
    if not qualitative:
        return
    os.makedirs(out_dir, exist_ok=True)
    n = len(qualitative)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i, ex in enumerate(qualitative):
        axes[i, 0].imshow(ex["image"], cmap="gray")
        axes[i, 0].set_title(f"input\ntrue={ex['label']}  pred={ex['pred_label']}", fontsize=9)
        axes[i, 1].imshow(ex["gt_mask"], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("ground truth mask" if ex["has_mask"] else "no mask (no_tumor)", fontsize=9)
        axes[i, 2].imshow(ex["pred_mask"], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("predicted mask", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "qualitative_examples.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved qualitative examples to {path}")


def main():
    image_size = (512, 512)
    base_ch = 32
    embed_dim = 256
    block_num = 4
    num_heads = 8    
    out_dir = 'report'

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_csv = "test_dataset_index.csv"
    test_ds = Brisc2025Dataset(test_csv, task="test", image_size=(512, 512))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_ds)} images")

    checkpoint = 'best_model.pt'
    model = load_model(checkpoint, device, image_size, base_ch,
                        embed_dim, block_num, num_heads)

    labels, preds, dice_overall, dice_per_class, qualitative = run_inference(
        model, test_loader, device)

    print("\n=== Classification ===")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (rows=true, cols=pred):", CLASS_NAMES)
    print(cm)

    print("\n=== Segmentation ===")
    mean_dice = float(np.mean(dice_overall)) if dice_overall else float("nan")
    print(f"Overall Dice (tumor-bearing images only): {mean_dice:.4f}  (n={len(dice_overall)})")
    for c in range(NUM_CLASSES):
        scores = dice_per_class[c]
        if scores:
            print(f"  {CLASS_NAMES[c]:>12}: Dice={np.mean(scores):.4f}  (n={len(scores)})")

    save_qualitative(qualitative, out_dir)

    report_path = os.path.join(out_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
        f.write(f"\nOverall Dice: {mean_dice:.4f} (n={len(dice_overall)})\n")
        for c in range(NUM_CLASSES):
            scores = dice_per_class[c]
            if scores:
                f.write(f"{CLASS_NAMES[c]}: Dice={np.mean(scores):.4f} (n={len(scores)})\n")
    print(f"\nSaved full report to {report_path}")


if __name__ == "__main__":
    main()
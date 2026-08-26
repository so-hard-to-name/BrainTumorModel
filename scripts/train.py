import torch
from torch.utils.data import DataLoader

from data_index import prepare_dataset_index
from src.brisc.dataset import Brisc2025Dataset
from src.brisc.models.multitask_model import MultiTaskModel
from src.brisc.losses import MultiTaskLoss

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n_seen = 0.0, 0, 0
    dice_scores = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)
            label = batch["tumor_type"].to(device)
            direction = batch["direction"].to(device)

            cls_logits, seg_logits = model(image, direction=direction)
            losses = loss_fn(cls_logits, seg_logits, label, mask, has_mask)
            total_loss += losses["total"].item() * image.size(0)

            pred_label = cls_logits.argmax(dim=1)
            correct += (pred_label == label).sum().item()
            n_seen += image.size(0)

            if has_mask.any():
                probs = torch.sigmoid(seg_logits[has_mask])
                preds = (probs > 0.5).float()
                gt = mask[has_mask]
                inter = (preds * gt).sum(dim(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
                dice = (2 * inter + 1e-6) / (union + 1e-6)
                dice_scores.extend(dice.tolist())

    avg_loss = total_loss / max(n_seen, 1)
    acc = correct / max(n_seen, 1)
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else float("nan")

    return avg_loss, acc, mean_dice

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main_dir = "brisc2025"
    prepare_dataset_index(main_dir, task="train")
    prepare_dataset_index(main_dir, task="test")

    train_csv = "train_dataset_index.csv"
    test_csv = "test_dataset_index.csv"

    train_ds = Brisc2025Dataset(train_csv, task="train", image_size=(512, 512))
    val_ds = Brisc2025Dataset(train_csv, task="val", image_size=(512, 512))
    test_ds = Brisc2025Dataset(test_csv, task="test", image_size=(512, 512))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, drop_last=False)
    val_loader  = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    epochs = 4
    image_size = 512
    max_tokens = (image_size // 16) ** 2

    model = MultiTaskModel(in_channels=1, num_classes=4, num_seg_ch=1,
                                base_ch=32, embed_dim=256, block_num=4, num_heads=8,
                                max_tokens=max_tokens, num_directions=4).to(device)
    loss_fn = MultiTaskLoss(cls_weight=1.0, seg_bce_weight=1.0, seg_dice_weight=1.0).to(device)

    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_dice = -1.0
    for epoch in range(1, 5):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)
            direction = batch["direction"].to(device, non_blocking=True)
            label = batch["tumor_type"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                cls_logits, seg_logits = model(image, direction=direction)
                losses = loss_fn(cls_logits, seg_logits, label, mask, has_mask)

            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses["total"].item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={losses['total'].item():.4f} "
                      f"(cls={losses['cls_loss'].item():.4f} "
                      f"bce={losses['seg_bce'].item():.4f} "
                      f"dice={losses['seg_dice'].item():.4f})")

        scheduler.step()
        val_loss, val_acc, val_dice = evaluate(model, test_loader, loss_fn, device)
        print(f"== epoch {epoch} done | train_loss={running_loss/len(train_loader):.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  saved new best checkpoint (dice={best_dice:.4f})")

    torch.save(model.state_dict(), "last_model.pt")


if __name__ == "__main__":
    main()
import torch

def set_warmup_lr(optimizer, base_lr, current_step, warmup_steps):
    if current_step >= warmup_steps:
        return
    lr = base_lr * (current_step + 1) / warmup_steps
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch,
                     base_lr, warmup_steps=0, max_grad_norm=1.0, log_every=50):
    model.train()
    running_loss = 0.0
    steps_per_epoch = len(loader)

    for step, batch in enumerate(loader):
        global_step = (epoch - 1) * steps_per_epoch + step
        set_warmup_lr(optimizer, base_lr, global_step, warmup_steps)

        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True)
        label = batch["tumor_type"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            cls_logits, seg_logits = model(image, direction=direction)
            losses = loss_fn(cls_logits, seg_logits, label, mask, has_mask)

        scaler.scale(losses["total"]).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += losses["total"].item()
        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch} step {step}/{steps_per_epoch} "
                  f"lr={current_lr:.2e} "
                  f"loss={losses['total'].item():.4f} "
                  f"(cls={losses['cls_loss'].item():.4f} "
                  f"bce={losses['seg_bce'].item():.4f} "
                  f"dice={losses['seg_dice'].item():.4f})")

    return running_loss / steps_per_epoch
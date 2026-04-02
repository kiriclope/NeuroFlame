import math

import torch
import torch.optim as optim

from contextlib import nullcontext

def maybe_zero_low_rank_grads(model, zero_grad):
    if zero_grad is None or not hasattr(model, "low_rank"):
        return

    u_grad = getattr(getattr(model.low_rank, "U", None), "grad", None)
    v_grad = getattr(getattr(model.low_rank, "V", None), "grad", None)

    if zero_grad == "all":
        if u_grad is not None:
            u_grad.zero_()
        if v_grad is not None:
            v_grad.zero_()
        return

    if u_grad is not None:
        u_grad[:, zero_grad] = 0
    if v_grad is not None:
        v_grad[:, zero_grad] = 0


def run_epoch(
    dataloader,
    model,
    loss_fn,
    optimizer=None,
    zero_grad=None,
    scaler=None,
    amp_enabled=False,
):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    use_amp = amp_enabled and str(model.device).startswith("cuda")
    context = torch.enable_grad() if is_training else torch.no_grad()

    if use_amp:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    with context:
        for X, y in dataloader:
            X = X.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                model(X)
                loss = loss_fn(model.readout, y)

            if is_training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    maybe_zero_low_rank_grads(model, zero_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    maybe_zero_low_rank_grads(model, zero_grad)
                    optimizer.step()

            batch_size = X.size(0)
            total_loss += loss.detach().float().item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return float("nan")

    return total_loss / total_samples


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    thresh=0.15,
    zero_grad=None,
    amp_enabled=True,
):
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    use_amp = amp_enabled and str(model.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            zero_grad=zero_grad,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            criterion,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )

        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Training Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}",
            flush=True,
        )

        if math.isnan(train_loss) or math.isnan(val_loss):
            print("Stopping: NaN loss", flush=True)
            break

        if val_loss > 300:
            print(f"Stopping: validation loss too high ({val_loss:.4f})", flush=True)
            break

        if train_loss < thresh and val_loss < thresh:
            print(
                f"Stopping: both losses below threshold "
                f"({train_loss:.4f}, {val_loss:.4f})", flush=True
            )
            break

    return history


def set_j_stp_trainable(model, trainable):
    model.J_STP.requires_grad = trainable


def freeze_for_dpa(model):
    set_j_stp_trainable(model, True)


def freeze_for_gonogo(model):
    set_j_stp_trainable(model, False)


def freeze_for_dual(model):
    set_j_stp_trainable(model, False)


def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, tuple(param.shape))

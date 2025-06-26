import torch


def save_ckpt(path, model, scaler, optimizer, scheduler, epoch):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    if scaler:
        state["grad_scaler"] = scaler.state_dict()
    torch.save(state, path)

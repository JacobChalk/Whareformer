from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def build_scheduler(config, optimizer, steps_per_epoch):
    scheduler_cfg = config.get('scheduler', {})
    if scheduler_cfg.get('scheduler_type') != 'linear_warmup+cosine':
        return None

    warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)
    eta_min = scheduler_cfg.get("eta_min", 1e-6)
    step_per_iter = scheduler_cfg.get('step_per_iter', True)
    total_epochs = config.get("epochs", 100)

    if step_per_iter:
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
    else:
        warmup_steps = warmup_epochs
        total_steps = total_epochs

    main_steps = total_steps - warmup_steps

    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=eta_min)
        ],
        milestones=[warmup_steps]
    ), step_per_iter
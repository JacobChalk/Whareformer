import numpy as np
import wandb 

def init_logging(logging_cfg, rank, model_cfg, batch_size):
    if logging_cfg.get('enable_wandb', False) and rank == 0:
        wandb.init(
            project=logging_cfg.get("project_name", "object-tracking"),
            name=logging_cfg.get("run_name", f"{model_cfg.get('name')}_experiment"),
            config={
                "model_type": model_cfg.get("name"),
                "depth": model_cfg['architecture']['depth'],
                "hidden_dim": model_cfg['architecture']['hidden_dim'],
                "batch_size": batch_size
            }
        )
        return True
    return False

def log_metrics(stage, metrics_dict, print_step=False, step=None, use_wandb=False, optimizer=None):
    if optimizer is not None:
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = lrs[0] if len(lrs) == 1 else lrs
        metrics_dict["lr"] = lr

    msg = f"{stage}"
    for key in ["loss", "accuracy", "precision", "recall", "f1", "balanced_accuracy",
                "weighted_precision", "weighted_recall", "weighted_f1", "lr"]:
        if metrics_dict is not None and key in metrics_dict:
            msg += f" | {key.replace('_', ' ').title()}: {metrics_dict[key]:.6f}"

    if print_step:
        print(msg)

    stage = stage.strip()

    if use_wandb and metrics_dict is not None:
        wandb_log_dict = {}

        for key, value in metrics_dict.items():
            if isinstance(value, (int, float, np.float32, np.float64)):
                wandb_log_dict[f"{stage}/{key}"] = value

        if step is not None:
            wandb_log_dict['epoch'] = step
        wandb.log(wandb_log_dict)
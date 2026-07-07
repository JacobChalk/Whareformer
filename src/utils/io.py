import torch
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_training_state(optimizer, epoch, path, scheduler=None):
    ensure_dir(os.path.dirname(path))
    state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
        
    torch.save(state, path)

def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save({k.replace('module.', ''): v for k, v in model.state_dict().items()}, path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_experiment_dir(base_path, model_config):
    def flatten_config(cfg, exclude_keys=None):
        exclude_keys = exclude_keys or set()
        items = []
        
        def recurse(d):
            for k, v in d.items():
                if k in exclude_keys:
                    continue
                if isinstance(v, dict):
                    recurse(v) 
                else:
                    if isinstance(v, float):
                        v = f"{v:.0e}" if v < 1e-2 and v != 0.0 else str(v)
                    items.append((k, v))
        
        recurse(cfg)
        return items

    exp_name = model_config['name']

    flat_items = flatten_config(model_config, exclude_keys={'name', 'logging', 'type', 'resume', 'grid_search'})
    for k, v in flat_items:
        exp_name += f'_{k}_{v}'

    full_path = os.path.join(base_path, exp_name)
    return full_path

def prepare_output_dir(cfg, model_config):
    out_root = cfg.get('paths.train_out_dir')
    run_name = model_config.get('training', {}).get('logging', {}).get('run_name')
    if run_name:
        out_dir = os.path.join(out_root, run_name)
    else:
        out_dir = get_experiment_dir(out_root, model_config)
    
    ensure_dir(out_dir)
    return out_dir
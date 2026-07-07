import torch

from .whareformer import Whareformer

def get_model(model_type, **kwargs):
    if model_type == "whareformer":
        return Whareformer(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

def init_model(model_config, device, is_distributed, local_rank):
    architecture = model_config.get('architecture', {})
    model = get_model(model_config.get('name'), **architecture).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    return model
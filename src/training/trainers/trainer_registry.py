TRAINER_REGISTRY = {}

def register_trainer(name):
    def decorator(cls):
        TRAINER_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def build_trainer(name, **kwargs):
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{name}' not found in registry. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[name](**kwargs)
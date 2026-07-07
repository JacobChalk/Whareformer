MATCHER_REGISTRY = {}

def register_matcher(name):
    def decorator(cls):
        MATCHER_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def build_matcher(name, params):
    name = name.lower()
    if name not in MATCHER_REGISTRY:
        raise ValueError(f"Matcher '{name}' not found in registry. Available: {list(MATCHER_REGISTRY.keys())}")
    return MATCHER_REGISTRY[name](name, params)
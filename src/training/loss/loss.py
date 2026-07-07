import torch.nn as nn


def get_loss(name: str):
    if name == "ce":
        return  nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-1)
    else:
        raise ValueError(f"Unknown loss function: {name}")
import os
import torch
import torch.distributed as dist

def parse_distributed_env():
    if all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return True, int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        return False, 0, 0, 1

def setup_ddp(rank):
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(rank)
    assert dist.is_initialized()
    
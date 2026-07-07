import torch.distributed as dist
import numpy as np
import argparse
import random
import wandb
import torch
import os

from pprint import pprint

from utils.io import save_model, save_training_state, load_model, prepare_output_dir
from utils.config import Config
from models import init_model

from training.utils.distributed import parse_distributed_env, setup_ddp
from training.utils.logging import init_logging, log_metrics
from training.utils.scheduler import build_scheduler
from training.data import load_training_data 
from training.trainers import build_trainer
from training.loss import get_loss

def generate_seed():
    return int.from_bytes(os.urandom(8), byteorder="big")

def set_seed(seed: int = 0):
    if seed < 0:
        seed = generate_seed()
    seed = seed % (2**32)
    print(f"Using Seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_training(trainer, train_loader, test_loader, epochs, out_dir, 
                 global_rank, enable_wandb, optimizer, model, is_distributed,
                 start_epoch=1, dagger_exit_epoch=-1):
    global_step = 0
    end_epoch = dagger_exit_epoch if dagger_exit_epoch != -1 else epochs
    for epoch in range(start_epoch, end_epoch + 1):
        if global_rank == 0:
            print(f"\nEpoch {epoch}/{epochs}")
        train_metrics, global_step = trainer.train_one_epoch(
            train_loader, log_interval=10, use_wandb=enable_wandb, global_step_start=global_step
        )
        if test_loader is not None:
            test_metrics = trainer.evaluate(test_loader)

        if global_rank == 0:
            log_metrics("Train", train_metrics, print_step=True, step=epoch, use_wandb=enable_wandb, optimizer=optimizer)
            if test_loader is not None:
                log_metrics("Test ", test_metrics, print_step=True, step=epoch, use_wandb=enable_wandb)
            save_model(model, os.path.join(out_dir, f"model_epoch{epoch}.pt"))

            save_training_state(
                optimizer, 
                epoch, 
                os.path.join(out_dir, f"training_state_epoch{epoch}.pt"), 
                trainer.scheduler
            )

    if is_distributed:
        dist.barrier()

def train(cfg):
    is_distributed, local_rank, global_rank, world_size = parse_distributed_env()
    if is_distributed:
        setup_ddp(local_rank)
        dist.barrier()

    gpus_per_node = torch.cuda.device_count() if is_distributed else 1
    num_nodes = max(1, world_size // gpus_per_node)

    print(f"[Rank {global_rank}] {num_nodes} node(s), {gpus_per_node} GPU(s)")

    try:
        model_config = cfg.get('model')
        if not model_config:
            raise ValueError("Missing model config!")
        
        pprint(model_config)

        training_config = model_config.get('training', {})
        training_seed = training_config.get('seed', -1)
        set_seed(training_seed)

        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        model = init_model(model_config, device, is_distributed, local_rank)
        pprint(model)

        if is_distributed:
            dist.barrier()

        batch_size = training_config.get('batch_size', 128)
        train_loader, test_loader = load_training_data(cfg, training_config, batch_size, world_size, global_rank, is_distributed)

        criterion = get_loss(training_config.get('loss', 'ce'))
        optimizer = torch.optim.AdamW(model.parameters(), 
                            lr=float(training_config.get('lr', 1e-3)), 
                            weight_decay=float(training_config.get('weight_decay', 1e-3))
                        )
        scheduler, step_per_iter = build_scheduler(training_config, optimizer, len(train_loader))

        resume_path = training_config.get('resume')
        start_epoch = 1
        if resume_path:
            model = load_model(model, resume_path, device=device)
            print("Resumed model from checkpoint.")
            
            state_path = resume_path.replace('model_epoch', 'training_state_epoch')
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=device)
                optimizer.load_state_dict(state['optimizer'])
                
                if 'scheduler' in state and scheduler is not None:
                    scheduler.load_state_dict(state['scheduler'])
                
                start_epoch = state['epoch'] + 1
                print(f"Successfully loaded optimizer state. Resuming from epoch {start_epoch}.")
            else:
                print(f"WARNING: No training state found at {state_path}. Optimizer starting from scratch.")

        if is_distributed:
            dist.barrier()

        trainer = build_trainer(
            model_config.get('name'),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_rank=global_rank,
            is_distributed=is_distributed,
            scheduler_step_per_batch=step_per_iter,
            scheduler=scheduler
        )

        out_dir = prepare_output_dir(cfg, model_config)
        enable_wandb = init_logging(training_config.get('logging', {}), global_rank, model_config, batch_size)

        if is_distributed:
            dist.barrier()

        run_training(
            trainer=trainer,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=training_config.get('epochs', 100),
            out_dir=out_dir,
            global_rank=global_rank,
            enable_wandb=enable_wandb,
            optimizer=optimizer,
            model=model,
            is_distributed=is_distributed,
            start_epoch=start_epoch,
            dagger_exit_epoch=training_config.get('dagger_exit_epoch', -1)
        )

        if enable_wandb and global_rank == 0:
            wandb.finish()

        if is_distributed:
            dist.barrier()
    finally:
        if is_distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/whareformer_config.yaml", help="Path to config file")  
    args = parser.parse_args()

    print('Loading Config')
    cfg = Config(args.config)
    train(cfg)
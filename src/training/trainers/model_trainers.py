import torch
import wandb

import torch.distributed as dist

from tqdm import tqdm

from .trainer_registry import register_trainer
from training.utils.metrics import compute_metrics


class ModelTrainer(object):
    def __init__(self, model, criterion, optimizer, device, global_rank, is_distributed,
                scheduler=None, scheduler_step_per_batch=False):
        self.model = model
        self.criterion = criterion
            
        self.optimizer = optimizer
        self.device = device
        self.global_rank = global_rank
        self.scheduler = scheduler
        self.is_distributed = is_distributed
        self.scheduler_step_per_batch = scheduler_step_per_batch

    def train_step(self, batch):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eval_step(self, batch):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute_metrics(self, targets, preds):
        return compute_metrics(targets, preds)

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def model_forward_pass(self, inputs, targets, masks=None):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _gather_distributed(self, tensor):
        local_size = torch.tensor([tensor.shape[0]], device=self.device)
        all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device)
                     for _ in range(dist.get_world_size())]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)

        if tensor.shape[0] < max_size:
            pad = torch.zeros(
                max_size - tensor.shape[0], *tensor.shape[1:],
                dtype=tensor.dtype, device=self.device
            )
            tensor = torch.cat([tensor, pad], dim=0)

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)

        return torch.cat([t[:s.item()] for t, s in zip(gathered, all_sizes)], dim=0)

    def train_one_epoch(self, dataloader, log_interval=10, use_wandb=False, global_step_start=0):
        self.set_train_mode()
        all_preds, all_targets = [], []
        total_loss, total_samples = 0, 0

        global_step = global_step_start

        if self.global_rank == 0:
            iterator = tqdm(dataloader, desc="Training  ")
        else:
            iterator = dataloader

        for batch in iterator:
            loss, preds, targets = self.train_step(batch)

            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()

            loss_val = loss.item()

            if use_wandb and global_step % log_interval == 0 and self.global_rank == 0:
                wandb.log({
                    "Train/iter_loss": loss_val,
                    "Train/lr": self.optimizer.param_groups[0]["lr"],
                    "Train/train_step": global_step
                }, step=global_step)

            global_step += 1
            all_preds.append(preds)
            all_targets.append(targets)

            total_loss += loss_val
            total_samples += 1

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if self.is_distributed:
            all_preds = self._gather_distributed(all_preds)
            all_targets = self._gather_distributed(all_targets)

        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()

        total_loss = torch.tensor(total_loss, device=self.device)
        total_samples = torch.tensor(total_samples, device=self.device)

        if self.is_distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        avg_loss = total_loss.item() / total_samples.item()

        if self.scheduler is not None and not self.scheduler_step_per_batch:
            self.scheduler.step()

        metrics = {}
        if self.global_rank == 0:
            metrics = self.compute_metrics(all_targets, all_preds)
            metrics["loss"] = avg_loss
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        if self.is_distributed:
            dist.barrier()

        return metrics, global_step

    def evaluate(self, dataloader):
        self.set_eval_mode()

        if len(dataloader) == 0:
            return {}

        all_preds, all_targets = [], []
        total_loss, total_samples = 0, 0

        if self.global_rank == 0:
            iterator = tqdm(dataloader, desc="Validating")
        else:
            iterator = dataloader

        for batch in iterator:
            loss, preds, targets = self.eval_step(batch)

            total_loss += loss.item()
            total_samples += 1

            all_preds.append(preds)
            all_targets.append(targets)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if self.is_distributed:
            all_preds = self._gather_distributed(all_preds)
            all_targets = self._gather_distributed(all_targets)

        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()

        total_loss = torch.tensor(total_loss, device=self.device)
        total_samples = torch.tensor(total_samples, device=self.device)

        if self.is_distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        avg_loss = total_loss.item() / total_samples.item()

        metrics = {}
        if self.global_rank == 0:
            metrics = self.compute_metrics(all_targets, all_preds)
            metrics["loss"] = avg_loss

        if self.is_distributed:
            dist.barrier()

        return metrics


@register_trainer("whareformer")
class WhareformerTrainer(ModelTrainer):
    def train_step(self, batch):
        inputs, masks, targets = batch
        inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        loss, logits = self.model_forward_pass(inputs, targets, masks)
        loss.backward()
        self.optimizer.step()
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        return loss, preds.detach(), targets.detach()

    def eval_step(self, batch):
        with torch.no_grad():
            inputs, masks, targets = batch
            inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)

            loss, logits = self.model_forward_pass(inputs, targets, masks)
            preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
            return loss, preds, targets

    def model_forward_pass(self, inputs, targets, masks=None):
        logits = self.model(inputs, masks)

        loss = self.criterion(logits, targets)

        return loss, logits
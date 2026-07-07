
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .track_dataset import TrackFeatureDataset
from utils.imbalance import compute_new_tracks


def collate_variable_detections(batch):
    """
    Args:
        batch: list of tuples (x_i, y_i)
            - x_i: Tensor [D_i x (T_i + 1) x C]  (+1 for detection row)
            - y_i: Tensor [D_i]
    Returns:
        padded_inputs: [D x (T_max + 1) x C]
        tracks_mask:   [D x T_max]         (Bool, True = PAD, False = VALID)
        labels:        [D]                 (Long Int Indices)
    """
    inputs, labels = zip(*batch)
    T_max_plus_1 = max(x.shape[1] for x in inputs)
    padded_inputs = []
    for x in inputs:
        pad_len = T_max_plus_1 - x.shape[1]
        x_padded = F.pad(x, (0, 0, 0, pad_len))
        padded_inputs.append(x_padded)
    padded_inputs = torch.cat(padded_inputs, dim=0)
    labels = torch.cat(labels, dim=0).long()

    T_max = T_max_plus_1 - 1
    tracks_per_sample = torch.tensor(
        [x.shape[1] - 1 for x in inputs for _ in range(x.shape[0])],
        dtype=torch.long
    )
    tracks_mask = torch.arange(T_max)[None, :] >= tracks_per_sample[:, None]  # (D, T_max)

    return padded_inputs, tracks_mask, labels


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.init_lmdb()

def get_track_dataset(gt_train_paths, test_paths,
                    batch_size: int = 128, seed=0, num_workers=8,
                    world_size=1, global_rank=0, is_distributed=False):
    gt_train_dir = os.path.dirname(gt_train_paths[0]) 
    gt_lmdb_path = os.path.join(os.path.dirname(gt_train_dir), 'train.lmdb')

    train_ds = TrackFeatureDataset(
        gt_manifest_paths=gt_train_paths, 
        gt_lmdb_path=gt_lmdb_path
    )

    if global_rank == 0:
        compute_new_tracks(train_ds)
        
    if is_distributed:
        dist.barrier()

    train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=global_rank
        ) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size // world_size,
        shuffle=False if is_distributed else True,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_variable_detections,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=True
    )
    
    if len(test_paths) > 0:
        test_dir = os.path.dirname(test_paths[0])
        test_lmdb_path = os.path.join(os.path.dirname(test_dir), 'test.lmdb')
        test_ds = TrackFeatureDataset(test_paths, lmdb_path=test_lmdb_path)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, 
                                    rank=global_rank, shuffle=False) if is_distributed else None
    
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size // world_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_variable_detections,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=True
        )
    else:
        test_loader = None

    return train_loader, test_loader

def get_deterministic_file_paths(directory, manifest_name="train_manifest.txt"):
    """
    Reads paths from a manifest file if it exists. The manifest is needed 
    to reproduce Whareformer results.
    """
    if not os.path.exists(directory):
        return []
        
    manifest_path = os.path.join(directory, manifest_name)

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        return [os.path.join(directory, f) for f in filenames]
    else:
        filenames = sorted(os.listdir(directory))
        return [os.path.join(directory, f) for f in filenames]

def load_training_data(cfg, training_config, batch_size, world_size, global_rank, is_distributed):
    gt_data_root = cfg.get('paths.train_data_dir')

    if not gt_data_root:
        raise ValueError("paths.train_data_dir is required")
    
    gt_train_dir = os.path.join(gt_data_root, "train")
    gt_train_paths = get_deterministic_file_paths(gt_train_dir, "train_manifest.txt")

    if os.path.exists(f"{gt_data_root}/test.lmdb"):
        test_paths = [f"{gt_data_root}/test/{f}" for f in os.listdir(f"{gt_data_root}/test")]
    else:
        test_paths = []
    
    workers = max(1, training_config.get('num_workers', 1) // torch.cuda.device_count())
    
    return get_track_dataset(
        gt_train_paths=gt_train_paths,
        test_paths=test_paths,
        batch_size=batch_size,
        seed=training_config.get("seed", 0),
        num_workers=workers,
        world_size=world_size,
        global_rank=global_rank,
        is_distributed=is_distributed
    )

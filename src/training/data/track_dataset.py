import numpy as np
import pickle
import random
import torch
import math
import lmdb
import os

from torch.utils.data import Dataset
from functools import lru_cache
from tqdm import tqdm

class LMDBFeatureLoader:
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(
            self.lmdb_path, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )

    @lru_cache(maxsize=8096)
    def get_feature(self, key: str, arr_dtype: np.dtype = np.float32) -> np.ndarray:
        with self.env.begin(write=False) as txn:
            byte_data = txn.get(key.encode('utf-8'))
            if byte_data is None:
                raise KeyError(f"Missing key in LMDB: {key}")
            return np.frombuffer(byte_data, dtype=arr_dtype)
        
    def get_features_from_keys(self, keys: list[str], arr_dtype: np.dtype = np.float32) -> np.ndarray:
        return np.stack([self.get_feature(key, arr_dtype) for key in keys])
    

class TrackFeatureDataset(Dataset):
    def __init__(self, gt_manifest_paths, gt_lmdb_path):
        self.gt_lmdb_path = gt_lmdb_path
        
        self.gt_loader = None
        
        print("Loading Ground Truth manifests...")
        self.gt_samples = self._load_manifests(gt_manifest_paths)

    def _load_manifests(self, file_paths):
        samples = []
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        for base_dir in tqdm(file_paths, leave=False):
            video_id = os.path.basename(base_dir.rstrip('/'))
            manifest_path = os.path.join(base_dir, f"{video_id}_manifest.pkl")
            with open(manifest_path, 'rb') as f:
                data = pickle.load(f)
                data = list(data.values())[0]
                for manifest in data:
                    if len(manifest['track_app_keys']) > 0:
                        samples.append(manifest)
        return samples

    def init_lmdb(self):
        """Initialises the LMDB environment for the current worker process."""
        self.gt_loader = LMDBFeatureLoader(self.gt_lmdb_path)

    def __len__(self):
        return len(self.gt_samples)

    def __getitem__(self, idx):
        if self.gt_loader is None:
            raise RuntimeError("LMDB loader not initialised.")
        manifest = self.gt_samples[idx]
        
        # Fetch detection features
        det_app = self.gt_loader.get_feature(manifest['det_app_key']).reshape(1, -1)
        det_loc = self.gt_loader.get_feature(manifest['det_loc_key']).reshape(1, -1)

        # Fetch track features
        track_app_keys = manifest['track_app_keys']
        track_loc_keys = manifest['track_loc_keys']
        
        T = len(track_app_keys)

        if T == 0:
            raise RuntimeError('Found T = 0 Sample!')

        det_feats = np.concatenate([det_app, det_loc], axis=-1)

        track_apps = self.gt_loader.get_features_from_keys(track_app_keys)
        track_locs = self.gt_loader.get_features_from_keys(track_loc_keys)
        track_feats = np.concatenate([track_apps, track_locs], axis=-1)

        features = np.concatenate([det_feats, track_feats], axis=0)

        x = torch.from_numpy(features)  # Shape: (T, C)
        y = torch.from_numpy(np.array([manifest['label'].argmax()])).long() # Shape: (1,)

        return x.unsqueeze(0), y

    def get_class_counts(self):
        # Returns 1 if it's a match to an existing track, 0 otherwise (new track)
        return [int(s['label'].argmax() > 0) for s in self.gt_samples]
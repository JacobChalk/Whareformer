import os
import time
import pickle
import gc
import torch
import random
import json
import numpy as np

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

class BaseFeatureExtractor:
    def __init__(self, video_id, cfg):
        self.video_id = video_id
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cfg = cfg
        
        self.output_dir = os.path.join(cfg.get('paths.observation_dir'), f"{video_id}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.object_annotations = {"video_annotations": []}
        self.save_dict_2D = {}
        self.save_dict_3D = {}
        
        self.data_loader = None 

        set_seed()

    def _format_time(self, seconds):
        if seconds < 0: return "-:--:--"
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0: return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
    
    def _check_existing(self):
        f_2d = os.path.exists(os.path.join(self.output_dir, f"2D_feat_masked.pkl"))
        f_3d = os.path.exists(os.path.join(self.output_dir, f"3D_feat_aligned.pkl"))
        f_mask = os.path.exists(os.path.join(self.output_dir, "object_annotations.json"))
        return f_2d and f_3d and f_mask

    def run(self):
        print(f"--- Starting processing for {self.video_id} ---")
        
        if self._check_existing():
            print("Files already exist. Skipping.")
            return

        total_batches = len(self.data_loader)
        if total_batches == 0:
            print("No batches to process.")
            return
            
        print(f"Found {total_batches} batches to process.")
        
        total_time_elapsed = 0
        t_batch_load_start = time.perf_counter()

        log_names = {'t_2d': '2D Feat', 't_3d': '3D Est', 't_sam': 'SAM/Crop'}

        for i, batch in enumerate(self.data_loader):
            t_batch_load_end = time.perf_counter()
            t_load = t_batch_load_end - t_batch_load_start
            
            t_process_start = time.perf_counter()
            timings = self._process_batch(batch)
            t_process = time.perf_counter() - t_process_start
            
            t_iter_total = t_load + t_process
            total_time_elapsed += t_iter_total

            avg_time_per_iter = total_time_elapsed / (i + 1)
            batches_remaining = total_batches - (i + 1)
            eta_seconds = batches_remaining * avg_time_per_iter
            eta_formatted = self._format_time(eta_seconds)
            
            if timings:
                timing_str = " | ".join([f"{log_names.get(k, k)}: {v:>6.2f}s" for k, v in timings.items()])
                
                log_str = (
                    f"Iter {i+1:>2}/{total_batches:<2} | "
                    f"Load: {t_load:>5.2f}s | "
                    f"{timing_str} | "
                    f"Iter: {t_iter_total:>6.2f}s | "
                    f"Elapsed: {self._format_time(total_time_elapsed):>6} | "
                    f"ETA: {eta_formatted:>6}"
                )

                log_len = len(log_str)
                percent_done = (i + 1) / total_batches
                percent_str = f" {percent_done * 100: >5.1f}%"
                bar_len = log_len - len(percent_str)
                filled_len = int(bar_len * percent_done)
                bar_str = '█' * filled_len + '─' * (bar_len - filled_len)
                progress_line = bar_str + percent_str

                print(f"{progress_line}\n{log_str}")

            t_batch_load_start = time.perf_counter()

        self.save_results()
        print(f"--- Finished processing for {self.video_id}. Total time: {self._format_time(total_time_elapsed)} ---")

    def _process_batch(self, batch):
        raise NotImplementedError("Subclasses must implement _process_batch")

    def save_results(self):
        path_annot = os.path.join(self.output_dir, "object_annotations.json")
        path_2d = os.path.join(self.output_dir, f"2D_feat_masked.pkl")
        path_3d = os.path.join(self.output_dir, f"3D_feat_aligned.pkl")
        
        with open(path_annot, 'w') as f:
            json.dump(self.object_annotations, f, indent=None)
        print(f"Saved Object Annotations to {path_annot}")

        with open(path_2d, 'wb') as f:
            pickle.dump(self.save_dict_2D, f)
        print(f"Saved 2D features to {path_2d}")

        with open(path_3d, 'wb') as f:
            pickle.dump(self.save_dict_3D, f)
        print(f"Saved 3D features to {path_3d}")

    def shutdown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
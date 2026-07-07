import pandas as pd
import numpy as np
import pickle
import random
import torch
import time
import lmdb
import os
import gc

from tqdm.auto import tqdm
from contextlib import nullcontext

from .tracker.detection import Detection
from .tracker.tracker import StandardTracker

from utils.data_loading import load_object_data
from utils.manifest_builder import build_feature_manifest_entry
from utils.distance import l2_normalise
from utils.evaluation import *

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

class VideoInstance:
    def __init__(self, cfg, video_id, tracking_params={}):
        self.cfg = cfg
        self.video_id = video_id
        self.dataset = self.cfg.get('dataset')
        self.split = 'train' if self.video_id in self.cfg.get('videos.train') else 'test'
        
        set_seed()
        self._setup_directories()
        self._load_metadata()
        self._load_features()
        self._setup_tracker(tracking_params)
        self._setup_lmdb()

    def _setup_directories(self):
        self.out_dir = self.cfg.get("paths.track_out_dir")
        self.observation_dir = self.cfg.get("paths.observation_dir")
        self.matrix_path = self.cfg.get("paths.train_data_dir")
        
        os.makedirs(os.path.join(self.out_dir, self.video_id), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'resource_result'), exist_ok=True)

    def _load_metadata(self):
        obj_identifier = 'name' if self.dataset == 'epic' else 'id'
        object_data_file = os.path.join(self.observation_dir, self.video_id, "object_annotations.json")
        self.object_data = load_object_data(object_data_file, obj_identifier=obj_identifier)
        self.video_data = self.cfg.get_video_data(self.video_id)
        self.fps = self.video_data.get("fps")

    def _load_features(self):
        app_feat_type = self.cfg.get("tracking.app_feat_type", 'masked')
        loc_feat_type = self.cfg.get("tracking.loc_feat_type", 'aligned')
        feat_2d_filename = f"2D_feat_{app_feat_type}"
        feat_3d_filename = f"3D_feat_{loc_feat_type}"

        pca_dim = self.cfg.get("tracking.pca_dim")
        self.use_pca = bool(pca_dim)
        if self.use_pca:
            feat_2d_filename = f"{feat_2d_filename}_PCA_{pca_dim}d"
            
        app_feat_path = f"{self.observation_dir}/{self.video_id}/{feat_2d_filename}.pkl"
        loc_feat_path = f"{self.observation_dir}/{self.video_id}/{feat_3d_filename}.pkl"

        with open(app_feat_path, 'rb') as file:
            self.app_feats = pickle.load(file)
            
        with open(loc_feat_path, 'rb') as file:
            self.loc_feats = pickle.load(file)

        assert self.app_feats.keys() == self.loc_feats.keys(), "Mismatch of frames for appearance and location features!"
        for k, v in self.app_feats.items():
            assert v.shape[0] == self.loc_feats[k].shape[0], f"Mismatch at frame {k}! Appearance: {v.shape[0]} Location: {self.loc_feats[k].shape[0]}"
            assert v.ndim == 2 and self.loc_feats[k].ndim == 2, f"Wrong dims at frame {k}! Appearance: {v.shape[0]} Location: {self.loc_feats[k].shape[0]}"

        self.tracked_frames = list(self.app_feats.keys())

    def _setup_tracker(self, tracking_params):
        tracking_params['app_dim'] = next(iter(self.app_feats.values())).shape[-1]
        self.tracking_params = tracking_params
        
        self.tracker = StandardTracker(tracking_params, self.fps)

        self.tracker_results = {}
        self.results_keys = ['track_ids', 'track_obj_names', 'track_bboxes', 'track_3d_locs']
        
        self.force_rewrite = self.cfg.get('tracking.force_rewrite', False)
        self.use_dense_eval = self.cfg.get('tracking.use_dense_eval', True)
        
        self.min_count = self.cfg.get('tracking.eval_min_count', 3)

    def _setup_lmdb(self):
        self.save_training_data = self.cfg.get('tracking.save_training_data')
        self.env = None
        if self.save_training_data and self.matrix_path:
            os.makedirs(f"{self.matrix_path}/{self.split}/{self.video_id}", exist_ok=True)
            lmdb_path = f"{self.matrix_path}/{self.split}/{self.video_id}/{self.video_id}.lmdb"
            self.env = lmdb.open(lmdb_path, map_size=50*1024**3)
            
        self.ctx = self.env.begin(write=True) if self.env is not None else nullcontext()

    def _prepare_detections(self, frame_name):
        objs, obj_bboxes = self.object_data.get(frame_name)
        app_features = self.app_feats[frame_name]
        loc_features = self.loc_feats[frame_name]

        if not self.use_pca:
            app_features = l2_normalise(app_features)

        detections = []
        frame_idx = int(frame_name.split('_')[-1])
        
        for det_idx in range(len(objs)):
            detection_data = {
                "frame": frame_idx,
                "detection_idx": det_idx,
                "obj_name": objs[det_idx],
                "bbox": obj_bboxes[det_idx],
                "app": app_features[det_idx],
                "loc": loc_features[det_idx]
            }
            detections.append(Detection(detection_data))
            
        return detections

    def _log_fps_usage(self, elapsed):
        num_frames = len(self.tracked_frames)
        avg_fps = num_frames / elapsed if elapsed > 0 else 0
        print(f"[{self.video_id}] Average FPS: {avg_fps:.2f}")

    def _save_lmdb_manifest(self, all_manifests):
        manifest_path = f"{self.matrix_path}/{self.split}/{self.video_id}/{self.video_id}_manifest.pkl"
        with open(manifest_path, "wb") as f:
            pickle.dump({self.video_id: all_manifests}, f)
            
        self.env.close()

    def track(self):
        output_path = os.path.join(self.out_dir, self.video_id, 'tracking_outputs.pkl')
        
        if not os.path.exists(output_path) or self.force_rewrite:
            start_time = time.perf_counter()

            feature_cache = {} 
            all_manifests = []

            with self.ctx as txn:
                for frame_name in tqdm(self.tracked_frames, desc= f"[{self.video_id}]", leave=False):
                    detections = self._prepare_detections(frame_name)
                    timestamp = float(frame_name.split('_')[-1]) / self.fps
                    frame_metadata = {
                        "frame_name": frame_name,
                        "frame_idx": int(frame_name.split('_')[-1]),
                        "timestamp": timestamp
                    }
                    
                    training_data = self.tracker.update(detections, frame_metadata)

                    if self.save_training_data and len(self.tracker.tracks) > 0 and training_data is not None:
                        raw_features, gt_matches, gt_unmatched_detections = training_data

                        manifest_entry = build_feature_manifest_entry(
                            gt_matches, gt_unmatched_detections, raw_features, 
                            self.video_id, frame_metadata["frame_idx"], 
                            txn, feature_cache
                        )
                        all_manifests.extend(manifest_entry)

                    self.tracker_results.setdefault(frame_name, {k: [] for k in self.results_keys})
                    for track in self.tracker.tracks:
                        if track.time_since_update == 0:
                            track_last_assignment, track_last_bbox, track_last_location = track.get_recent_assignment()
                            self.tracker_results[frame_name]['track_ids'].append(track.track_id)
                            self.tracker_results[frame_name]['track_obj_names'].append(track_last_assignment)
                            self.tracker_results[frame_name]['track_bboxes'].append(track_last_bbox)
                            self.tracker_results[frame_name]['track_3d_locs'].append(track_last_location)

            elapsed = time.perf_counter() - start_time
            self._log_fps_usage(elapsed)

            if self.save_training_data:
                self._save_lmdb_manifest(all_manifests)

            with open(output_path, 'wb') as f:
                pickle.dump(self.tracker_results, f)

        if not self.tracker_results:
            with open(output_path, 'rb') as f:
                self.tracker_results = pickle.load(f)
        self.evaluate_tracker()

    def evaluate_tracker(self):
        num_frames = self.video_data.get("num_frames")

        timescales = [self.fps * 5 * i for i in range(145)]
        distance_threhsolds = [0.3]

        obj_sets, obj_track_ids, obj_locs, track_locs, track_lifespans = get_tracking_data(self.tracker_results)

        if self.use_dense_eval:
            key_frames = get_dense_key_frames(obj_sets, min_count=self.min_count)
        else:
            key_frames = get_sparse_key_frames(obj_sets, fps=self.fps)

        results_df = {'n': []}    
        for dist_thresh in distance_threhsolds:
            results_df[f'r={dist_thresh}_pcl'] = []
            results_df[f'r={dist_thresh}_correct'] = []
            results_df[f'r={dist_thresh}_total'] = []

        for timescale in tqdm(timescales, desc=f"[{self.video_id}] Evaluating", position=0, leave=True):
            results_df['n'].append(int(timescale / self.fps))
            for dist_thresh in distance_threhsolds:
                accuracy, correct, total = get_timescale_accuracy(
                    key_frames,
                    timescale, dist_thresh,
                    obj_track_ids, obj_locs,
                    track_locs, track_lifespans,
                    final_frame=f"frame_{num_frames:010d}",
                    dense_eval=self.use_dense_eval
                )
                results_df[f'r={dist_thresh}_pcl'].append(accuracy)
                results_df[f'r={dist_thresh}_correct'].append(correct)
                results_df[f'r={dist_thresh}_total'].append(total)
            
        output_path = os.path.join(self.out_dir, self.video_id, 'results.csv')
        pd.DataFrame.from_dict(results_df).to_csv(output_path, index=False)

    def shutdown(self):
        try:
            print(f"[{self.video_id}] Releasing memory...")
            del self.object_data
            del self.loc_feats
            del self.app_feats
            del self.video_data

            if hasattr(self, 'tracker'):
                self.tracker.clear()
                del self.tracker

            if hasattr(self, 'tracker_results'):
                self.tracker_results.clear()
                del self.tracker_results

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[{self.video_id}] Memory released successfully.")
        except Exception as e:
            print(f"[{self.video_id}] Error during shutdown: {e}")
import os
import torch
import numpy as np
import time
import json
import subprocess

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import feature_extraction.utils.models as models
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor
from feature_extraction.datasets.epic_dataset import EPICFrameDataset
from feature_extraction.utils.loaders import ThreadPoolDataLoader, ObjectCountBatchSampler
from feature_extraction.utils.epic_utils import align_metric_and_scene_depth, lift_object_centroid, compute_metric_scale

class EPICFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, video_id, cfg):
        super().__init__(video_id, cfg)

        print(f"Loading 2D and 3D models for {video_id}...")
        self.model_2d, self.transform_2d, self.forward_fn_2d = models.load_2d_model(
            cfg.get('feature_extraction.model_2d'), self.device
        )

        self.model_depth, self.processor_depth, forward_fn_3d = models.load_depth_model(self.device)
        self.forward_fn_3d = lambda img: forward_fn_3d(
            img, self.model_depth, self.processor_depth, self.device, (456, 256)
        )
        self.depth_sub_batch_size = cfg.get('feature_extraction.depth_sub_batch_size')

        participant_id = video_id.split('_')[0]
        frames_path = os.path.join(
            cfg.get('paths.epic_frames_dir'), participant_id, 'rgb_frames', video_id
        )
        reconstruction_path = os.path.join(
            cfg.get('paths.epic_fields_dir'), 'sparse', video_id, 'sparse', '0'
        )
        scene_path = os.path.join(cfg.get('paths.scene_dir'), video_id)

        metric_scene_scale = self._resolve_metric_scale(
                                    video_id, reconstruction_path, 
                                    frames_path, scene_path
                                )
        mesh_path = self._resolve_mesh(reconstruction_path, frames_path, scene_path)

        frame_dataset = EPICFrameDataset(video_id, cfg, metric_scene_scale=metric_scene_scale)
        self.camera_intrinsics = frame_dataset.camera_intrinsics

        self.pyrender_scene = models.PyrenderScene(
            mesh_path, self.camera_intrinsics, scale_factor=metric_scene_scale
        )

        batch_sampler = ObjectCountBatchSampler(frame_dataset, target_object_count=cfg.get('feature_extraction.target_batch_size'))
        
        self.num_workers = cfg.get('feature_extraction.num_workers', 8)
        worker_fn = partial(
            frame_dataset.process_item, 
            transform_2d=self.transform_2d, 
        )
        self.data_loader = ThreadPoolDataLoader(
            dataset=frame_dataset,
            batch_sampler=batch_sampler,
            collate_fn=frame_dataset.collate_fn,
            worker_fn=worker_fn,
            num_workers=self.num_workers
        )
        self._alignment_executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        print(f"Initialisation complete for {video_id}.")

    def _resolve_metric_scale(self, video_id, reconstruction_path, 
                              frames_path, scene_path) -> float:
        cache_path = os.path.join(scene_path, 'metric_scale.json')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                scale = json.load(f)['scale']
            print(f"[Scale] Loaded cached metric scale: {scale:.10f}")
            return scale

        scale = compute_metric_scale(
            reconstruction_path=reconstruction_path,
            forward_fn_3d=self.forward_fn_3d,
            frames_path=frames_path,
            batch_size=self.depth_sub_batch_size
        )
        os.makedirs(scene_path, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'video_id': video_id, 'scale': scale}, f, indent=2)
        return scale
    
    def _resolve_mesh(self, reconstruction_path, frames_path, scene_path):
        mesh_path = os.path.join(scene_path, 'mvs_mesh.ply')
        if not os.path.exists(mesh_path):
            print(f"[{self.video_id}] MVS mesh not found. Triggering reconstruction script...")
            
            current_dir = Path(__file__).resolve().parent
            script_path = current_dir.parent / 'reconstruction' / 'reconstruct_mesh.sh'
            
            if not script_path.exists():
                raise FileNotFoundError(f"Cannot locate MVS script at: {script_path}")
            
            try:
                subprocess.run(
                    ["bash", str(script_path), self.video_id, reconstruction_path, frames_path, scene_path],
                    check=True,   
                    text=True,
                    capture_output=False
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MVS reconstruction failed for {self.video_id} with exit code {e.returncode}")
            
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Script completed successfully, but mesh is missing: {mesh_path}")

        return mesh_path

    def _align_and_lift_frame(self, frame_data_tuple):
        frame_name, camera_pose, bboxes, object_masks, metric_depth, masked_regions, scene_depth = frame_data_tuple

        alignment_mask = ((~masked_regions) & (scene_depth > 1e-6))
        aligned_depth = align_metric_and_scene_depth(metric_depth, scene_depth, alignment_mask)

        lifted_locations = lift_object_centroid(
                                            object_masks,
                                            bboxes,
                                            camera_pose,
                                            aligned_depth,
                                            self.camera_intrinsics,
                                            (854, 480),
                                        )
        return (frame_name, lifted_locations)

    def _process_batch(self, batch):
        if batch is None: return None
        timings = {}

        t_start = time.perf_counter()
        
        for i, frame_name in enumerate(batch['frame_names']):
            self.object_annotations["video_annotations"].append(batch['frame_annotations'][i])

        object_crops = batch['object_crops'].to(self.device, non_blocking=True)

        with torch.no_grad():
            appearance_features = self.forward_fn_2d(self.model_2d, object_crops).cpu().numpy()

        current_idx = 0
        for frame_name, num_objs in zip(batch['frame_names'], batch['num_objs_per_frame']):
            self.save_dict_2D[frame_name] = appearance_features[current_idx : current_idx + num_objs].astype(np.float32)
            current_idx += num_objs
        timings['t_2d'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        raw_frames_full_batch = batch['raw_frames']
        total_frames = raw_frames_full_batch.shape[0]
        depth_maps_list = [] 

        for start_idx in range(0, total_frames, self.depth_sub_batch_size):
            end_idx = min(start_idx + self.depth_sub_batch_size, total_frames)
            depth_maps_list.append(self.forward_fn_3d(raw_frames_full_batch[start_idx:end_idx]))
            
        depth_maps = torch.cat(depth_maps_list, dim=0).cpu().numpy()
        scene_depths = [self.pyrender_scene.get_scene_depth(pose) for pose in batch['camera_poses']]

        frame_data_tuples = zip(
            batch['frame_names'],
            batch['camera_poses'],
            batch['bboxes'],
            batch['object_masks'],
            depth_maps,
            batch['masked_frames'],
            scene_depths
        )
        
        results = list(self._alignment_executor.map(self._align_and_lift_frame, frame_data_tuples))
        
        for frame_name, locations in results:
            self.save_dict_3D[frame_name] = locations.astype(np.float32)
        timings['t_3d'] = time.perf_counter() - t_start

        return timings

    def shutdown(self):
        self._alignment_executor.shutdown(wait=False)
        del self.model_2d, self.model_depth, self.pyrender_scene, self.data_loader
        super().shutdown()
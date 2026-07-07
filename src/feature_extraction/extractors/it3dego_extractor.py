import os
import json
import time
import torch
import numpy as np

import feature_extraction.utils.models as models
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor
from feature_extraction.datasets.it3dego_dataset import IT3DEgoFrameDataset
from feature_extraction.utils.loaders import ThreadPoolDataLoader, ObjectCountBatchSampler
from feature_extraction.utils.common import create_object_crop

class IT3DEgoFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, video_id, cfg):
        super().__init__(video_id, cfg)

        pose_path = os.path.join(cfg.get('paths.it3dego_root'), "raw_videos", self.video_id, "depth_ahat_pose.json")
        with open(pose_path, 'r') as f:
            self.all_poses = json.load(f)

        vid_num = int(video_id.split('_')[1])
        calib_folder = "calibrations_vid1_17" if vid_num <= 17 else "calibrations_vid18_50"
        calib_base = os.path.join(cfg.get('paths.it3dego_root'), "calibrations", calib_folder)

        self._load_hl2ss_calibrations(calib_base)

        # Parse stationary intervals and the pre-computed 3D centers
        center_path = os.path.join(cfg.get('paths.it3dego_root'), "annotations", self.video_id, "3d_center_annot.txt")
        self.stationary_intervals = {}
        if os.path.exists(center_path):
            with open(center_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        start_ts, end_ts = int(parts[0]), int(parts[1])
                        obj_id = parts[2]
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        
                        if obj_id not in self.stationary_intervals:
                            self.stationary_intervals[obj_id] = []
                            
                        self.stationary_intervals[obj_id].append(
                            (start_ts, end_ts, np.array([x, y, z], dtype=np.float32))
                        )

        print(f"Loading Models for {video_id}...")
        self.model_2d, self.transform_2d, self.forward_fn_2d = models.load_2d_model(
            cfg.get('feature_extraction.model_2d'), self.device
        )
        self.seg_model, self.seg_model_processor, self.forward_fn_seg = models.load_instance_segmentation_model('sam2', self.device)
        self.seg_model_sub_batch_size = self.cfg.get('feature_extraction.segmentation_sub_batch_size', 2)

        dataset = IT3DEgoFrameDataset(video_id, cfg)
        sampler = ObjectCountBatchSampler(dataset, target_object_count=cfg.get('feature_extraction.target_batch_size'))
        self.data_loader = ThreadPoolDataLoader(
            dataset=dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn,
            worker_fn=dataset.process_item, num_workers=cfg.get('feature_extraction.num_workers')
        )

        self.object_annotations = {"video_annotations": []}

    def _load_hl2ss_calibrations(self, calib_base):
        """Loads IT3DEgo's calibrations and precompute the hl2ss projection matrices."""
        self.ahat_ext = np.fromfile(os.path.join(calib_base, "rm_depth_ahat", "extrinsics.bin"), dtype=np.float32).reshape(4, 4)
        ahat_uv2xy = np.fromfile(os.path.join(calib_base, "rm_depth_ahat", "uv2xy.bin"), dtype=np.float32).reshape(512, 512, 2)
        ahat_scale = np.fromfile(os.path.join(calib_base, "rm_depth_ahat", "scale.bin"), dtype=np.float32)[0]
        
        pv_ext = np.fromfile(os.path.join(calib_base, "personal_video", "extrinsics.bin"), dtype=np.float32).reshape(4, 4)
        pv_int = np.fromfile(os.path.join(calib_base, "personal_video", "intrinsics.bin"), dtype=np.float32).reshape(4, 4)

        self.ahat_to_pv_image = np.linalg.inv(self.ahat_ext) @ pv_ext @ pv_int

        self.xy1 = np.concatenate((ahat_uv2xy, np.ones((512, 512, 1), dtype=np.float32)), axis=-1)
        self.scale_map = np.linalg.norm(self.xy1, axis=2) * (ahat_scale / 4.0)

    def _compute_reproj_xyz_map(self, raw_depth):
        """Replicates compute_depth_scale_map_wrapper natively in NumPy."""
        safe_scale = np.where(self.scale_map == 0, 1e-6, self.scale_map)
        depth = raw_depth / safe_scale
        
        xyz = self.xy1 * depth[:, :, np.newaxis]
        xyz1 = np.concatenate((xyz, np.ones((512, 512, 1), dtype=np.float32)), axis=-1)

        uvw = xyz1 @ self.ahat_to_pv_image[:, 0:3]
        
        w_safe = np.where(uvw[..., 2:3] == 0, 1e-6, uvw[..., 2:3])
        uv = uvw[..., 0:2] / w_safe

        reproj_xyz_map = np.zeros((720, 1280, 3), dtype=np.float32)
        u = uv[..., 0].astype(int).reshape(-1)
        v = uv[..., 1].astype(int).reshape(-1)

        keep = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        reproj_xyz_map[v[keep], u[keep], :] = xyz.reshape(-1, 3)[keep]

        return reproj_xyz_map

    def _pv_bbox2depth_unproj(self, bbox, reproj_xyz_map, depth_pose):
        start_x, start_y, stop_x, stop_y = map(int, bbox)
        
        start_x, start_y = max(0, start_x), max(0, start_y)
        stop_x, stop_y = min(1280, stop_x), min(720, stop_y)
        
        if start_x >= stop_x or start_y >= stop_y:
            return [np.nan, np.nan, np.nan]

        xyz = reproj_xyz_map[start_y:stop_y, start_x:stop_x, :]
        y_idx, x_idx = xyz[..., 2].nonzero()

        # Follow IT3DEgos' logic: if no valid depth, skip (return NaN)
        if len(x_idx) == 0:
            return [np.nan, np.nan, np.nan]

        c_y, c_x = int((stop_y - start_y) / 2.0), int((stop_x - start_x) / 2.0)
        sel_idx = (np.abs(y_idx - c_y) + np.abs(x_idx - c_x)).argmin()
        valid_y, valid_x = y_idx[sel_idx], x_idx[sel_idx]
        
        det_tmp = xyz[valid_y, valid_x]

        matrix_t = np.linalg.inv(self.ahat_ext) @ np.array(depth_pose)
        det_tmp_homo = np.append(det_tmp, 1.0).reshape(1, 4)
        world_pt = det_tmp_homo @ matrix_t

        return world_pt.squeeze()[:3].tolist()

    def _process_batch(self, batch):
        if batch is None: return None
        timings = {}

        t_start = time.perf_counter()
        all_crops, masks_batch = [], []
        
        frames, bboxes = batch['frames'], batch['bboxes']
        with torch.no_grad():
            for start_idx in range(0, len(frames), self.seg_model_sub_batch_size):
                end_idx = start_idx + self.seg_model_sub_batch_size
                sub_imgs = frames[start_idx:end_idx]
                sub_boxes = bboxes[start_idx:end_idx]
                sub_masks = self.forward_fn_seg(self.seg_model, self.seg_model_processor, sub_imgs, sub_boxes)
                masks_batch.extend(sub_masks)
        timings['t_sam'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        valid_frame_names = []
        valid_num_objs_per_frame = []

        for i, (ts, frame_name) in enumerate(zip(batch['timestamps'], batch['frame_names'])):
            frame = frames[i]
            depth_map = batch['depth_maps'][i]
            frame_annotations = {"frame_name": frame_name, "annotations": []}
            
            frame_3d = []
            valid_objs_count = 0
            ts_int = int(ts)

            depth_pose = self.all_poses.get(ts)
            reproj_xyz_map = self._compute_reproj_xyz_map(depth_map)

            for j, bbox in enumerate(bboxes[i]):
                obj_id = str(batch['obj_ids'][i][j])
                
                obj_3d = self._pv_bbox2depth_unproj(bbox, reproj_xyz_map, depth_pose)
                    
                is_stationary = False
                for start_ts, end_ts, _ in self.stationary_intervals.get(obj_id, []):
                    if start_ts <= ts_int <= end_ts:
                        is_stationary = True
                        break
                        
                if np.isnan(obj_3d).any() and not is_stationary:
                    continue

                mask = (masks_batch[i][j, 0] > 0).cpu().numpy().astype(np.uint8)
                
                frame_annotations["annotations"].append({
                    "id": obj_id, 
                    "name": batch['obj_names'][i][j],
                    "bounding_box": list(map(int, bbox))
                })
                
                crop = create_object_crop(
                    frame, bbox, input_mask=mask, 
                    scale_frame_to_mask=False
                )
                all_crops.append(self.transform_2d(crop))
                
                frame_3d.append(obj_3d)
                valid_objs_count += 1

            if valid_objs_count == 0:
                continue

            self.object_annotations["video_annotations"].append(frame_annotations)
            self.save_dict_3D[frame_name] = np.array(frame_3d, dtype=np.float32)
            valid_frame_names.append(frame_name)
            valid_num_objs_per_frame.append(valid_objs_count)

        if all_crops:
            object_crops = torch.stack(all_crops).to(self.device)
            with torch.no_grad():
                appearance_features = self.forward_fn_2d(self.model_2d, object_crops).cpu().numpy()
            
            current_idx = 0
            for frame_name, num_objs in zip(valid_frame_names, valid_num_objs_per_frame):
                self.save_dict_2D[frame_name] = appearance_features[current_idx : current_idx + num_objs].astype(np.float32)
                current_idx += num_objs
                
        timings['t_2d'] = time.perf_counter() - t_start
        return timings

    def save_results(self):
        frame_name_to_ts = {v: int(k) for k, v in self.data_loader.dataset.ts_to_frame_name.items()}
        
        cleaned_object_annotations = {"video_annotations": []}
        
        for frame_data in self.object_annotations["video_annotations"]:
            frame_name = frame_data['frame_name']
            
            ts_int = frame_name_to_ts.get(frame_name)
            if ts_int is None: 
                continue
            
            valid_indices = []
            cleaned_annotations = []
            
            for obj_idx, annot in enumerate(frame_data["annotations"]):
                obj_id = str(annot["id"])
                
                for start_ts, end_ts, center_3d in self.stationary_intervals.get(obj_id, []):
                    if start_ts <= ts_int <= end_ts:
                        self.save_dict_3D[frame_name][obj_idx] = center_3d
                        break
                
                if not np.isnan(self.save_dict_3D[frame_name][obj_idx]).any():
                    valid_indices.append(obj_idx)
                    cleaned_annotations.append(annot)

            if len(valid_indices) == 0:
                self.save_dict_2D.pop(frame_name, None)
                self.save_dict_3D.pop(frame_name, None)
                continue
                
            new_frame_data = {"frame_name": frame_name, "annotations": cleaned_annotations}
            cleaned_object_annotations["video_annotations"].append(new_frame_data)
            
            self.save_dict_2D[frame_name] = self.save_dict_2D[frame_name][valid_indices]
            self.save_dict_3D[frame_name] = self.save_dict_3D[frame_name][valid_indices]

        self.object_annotations = cleaned_object_annotations
        super().save_results()

    def shutdown(self):
        del self.model_2d, self.seg_model, self.data_loader
        super().shutdown()
import os
import json
import numpy as np
import torch
import cv2

from feature_extraction.datasets.base_dataset import BaseFeatureDataset
from feature_extraction.utils.common import create_object_crop
from feature_extraction.utils.epic_utils import colmap_pose_to_c2w

class EPICFrameDataset(BaseFeatureDataset):
    def __init__(self, video_id, cfg, metric_scene_scale=1.0):
        super().__init__()
        self.video_id = video_id
        self.cfg = cfg
        participant_id = video_id.split('_')[0] if '_' in video_id else video_id.split('-')[0]
        self.frames_path = os.path.join(cfg.get('paths.epic_frames_dir'), participant_id, 'rgb_frames', video_id)
        
        self.multiple_instances = cfg.get('multiple_instances', {}).get(video_id, [])

        with open(os.path.join(cfg.get('paths.epic_fields_dir'), 'poses', f'{self.video_id}.json'), 'r') as f:
            raw_poses = json.load(f)

        cam_data = raw_poses['camera']
        fx, fy, cx, cy = cam_data['params'][:4]
        self.camera_intrinsics = [cam_data['width'], cam_data['height'], fx, fy, cx, cy]

        self.poses = {}
        for frame_name, vals in raw_poses["images"].items():
            qvec = np.array(vals[:4])
            tvec = np.array(vals[4:])
            self.poses[frame_name.replace('.jpg', '')] = colmap_pose_to_c2w(qvec, tvec, metric_scene_scale)

        with open(os.path.join(cfg.get('paths.visor_dir'), f'{self.video_id}_interpolations.json'), 'r') as f:
            masks_json = json.load(f)
            
        with open(cfg.get('paths.visor_frame_mapping'), 'r') as f:
            frame_mapping = json.load(f)

        self.masks, self.obj_counts = self._load_object_masks(masks_json['video_annotations'], frame_mapping)

        self.frame_list = sorted([k for k in self.masks if k in self.poses])

        self.poses = {k: self.poses[k] for k in self.frame_list}
        self.masks = {k: self.masks[k] for k in self.frame_list}
        self.obj_counts = {k: self.obj_counts[k] for k in self.frame_list}
        
        print(f"EPICFrameDataset initialised for {self.video_id} with {len(self.frame_list)} frames.")

    def _load_object_masks(self, data, frame_mapping):
        obj_masks = {}
        obj_counts = {}

        for entry in data:
            image_path = entry['image']['image_path']
            visor_frame_name = os.path.splitext(os.path.basename(image_path))[0]
            visor_frame_name = visor_frame_name.replace(f"{self.video_id}_", "")
            frame_name = frame_mapping.get(visor_frame_name, visor_frame_name)

            obj_list, bbox_list, segments_list = [], [], []
            alignment_segments = []

            for annotation in entry.get('annotations', []):
                obj_name = annotation['name']
                segments = annotation['segments']
                
                if not segments:
                    continue

                mask_points = [item for sublist in segments for item in sublist]
                if not mask_points:
                    continue
                
                if obj_name in ['left hand', 'right hand'] or obj_name in self.multiple_instances:
                    alignment_segments.append(segments)
                    continue

                mask_id = annotation['id']
                x_coords = [p[0] for p in mask_points]
                y_coords = [p[1] for p in mask_points]
                bbox = [min(x_coords), min(y_coords), max(x_coords) + 1, max(y_coords) + 1]

                obj_list.append((obj_name, mask_id))
                bbox_list.append(bbox)
                segments_list.append(segments)

            if len(bbox_list) > 0:
                if frame_name not in obj_masks:
                    obj_masks[frame_name] = (obj_list, bbox_list, segments_list, alignment_segments)
                    obj_counts[frame_name] = len(obj_list)
                else:
                    current_obj, current_bbox, current_seg, current_align = obj_masks[frame_name]
                    
                    # We use the object name as the deduplication key because IDs are per-mask, not per object-instance. It's the best we can do in VISOR
                    existing_objs = {o[0] for o in current_obj} 
                    
                    merged_count = len(current_bbox)
                    for j, obj in enumerate(obj_list):
                        if obj[0] not in existing_objs:
                            current_obj.append(obj)
                            current_bbox.append(bbox_list[j])
                            current_seg.append(segments_list[j])
                            merged_count += 1
                            
                    current_align.extend(alignment_segments)
                    
                    obj_masks[frame_name] = (current_obj, current_bbox, current_seg, current_align)
                    obj_counts[frame_name] = merged_count
        
        return obj_masks, obj_counts

    def __getitem__(self, idx):
        frame_name = self.frame_list[idx]

        frame_path = os.path.join(self.frames_path, f"{frame_name}.jpg")
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

        camera_pose = self.poses[frame_name]
        
        objs, bbs, segments, align_segments = self.masks[frame_name]

        return {
            'video_id': self.video_id,
            'frame_name': frame_name,
            'frame': frame,
            'camera_pose': camera_pose,
            'object_names': objs,
            'bboxes': bbs,
            'segments': segments,
            'align_segments': align_segments
        }

    @staticmethod
    def process_item(data, transform_2d=None):
        if data is None:
            return None
            
        frame_name = data['frame_name']
        frame = data['frame']
        camera_pose = data['camera_pose']
        obj_names = data['object_names']
        bboxes = data['bboxes']
        segments = data['segments']
        align_segments = data['align_segments']
        num_objs = len(bboxes)

        frame_h, frame_w = frame.shape[:2]
        mask_w, mask_h = (854, 480)

        frame_annotations = {
            "frame_name": frame_name,
            "annotations": []
        }
        
        all_object_crops = []
        object_masks = []
        for i, (bbox, segs) in enumerate(zip(bboxes, segments)):
            frame_annotations["annotations"].append({
                "id": obj_names[i][1], 
                "name": obj_names[i][0],
                "bounding_box": list(map(int, bbox))
            })

            obj_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            polygons = [np.array(p, dtype=np.int32) for p in segs]
            if polygons:
                cv2.fillPoly(obj_mask, polygons, color=1)
            object_masks.append(obj_mask.astype(bool))

            crop = create_object_crop(frame, bbox, input_mask=obj_mask)
            all_object_crops.append(transform_2d(crop))

        stacked_crops = torch.stack(all_object_crops)

        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        combined_mask_frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        all_segments_for_depth = segments + align_segments
        
        scale_x = frame_w / mask_w
        scale_y = frame_h / mask_h
        
        all_polygons_in_frame = [
            np.array([[int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))] for pt in p], dtype=np.int32)
            for segs_for_one_obj in all_segments_for_depth 
            for p in segs_for_one_obj
        ]
        
        if all_polygons_in_frame:
            cv2.fillPoly(combined_mask_frame, all_polygons_in_frame, color=1)
            
        combined_mask_frame = combined_mask_frame.astype(bool)
        
        return (
            frame_name,
            num_objs,
            bboxes,
            camera_pose,
            stacked_crops,
            frame_tensor,
            combined_mask_frame,
            object_masks,
            frame_annotations
        )
        
    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        
        unzipped_batch = list(zip(*batch))
        
        stacked_crops = unzipped_batch[4]
        if stacked_crops[0] is not None:
            object_crops = torch.cat(stacked_crops).pin_memory()
        else:
            object_crops = None
        
        return {
            'frame_names': list(unzipped_batch[0]),
            'num_objs_per_frame': list(unzipped_batch[1]),
            'bboxes': list(unzipped_batch[2]),
            'camera_poses': list(unzipped_batch[3]),
            'object_crops': object_crops,
            'raw_frames': torch.stack(unzipped_batch[5]).pin_memory(),
            'masked_frames': list(unzipped_batch[6]),
            'object_masks': list(unzipped_batch[7]),
            'frame_annotations': list(unzipped_batch[8])
        }
import os
import cv2
import json
import torch
import numpy as np
import decord

from feature_extraction.datasets.base_dataset import BaseFeatureDataset
from feature_extraction.utils.common import create_object_crop

class HDEPICFrameDataset(BaseFeatureDataset):
    def __init__(self, video_id, cfg):
        super().__init__()
        self.video_id = video_id
        self.cfg = cfg
        pid = video_id.split('-')[0]
        self.video_path = os.path.join(cfg.get('paths.video_dir'), pid, f"{video_id}.mp4")
        
        self.video_reader = decord.VideoReader(self.video_path, ctx=decord.cpu(0))
        
        mask_base = cfg.get('paths.masks_dir')
        self.unattached_dir = os.path.join(mask_base, 'unattached_masks', video_id)
        self.masks_dir = os.path.join(mask_base, 'masks', video_id)
        
        with open(cfg.get('paths.assoc_info'), 'r') as f:
            self.assoc_info = json.load(f).get(video_id, {})
        with open(cfg.get('paths.frame_info'), 'r') as f:
            self.frame_info = json.load(f).get(video_id, {})

        self.frames_data = self._build_frame_registry()
        self.frame_list = sorted(list(self.frames_data.keys()))
        self.obj_counts = {f_num: len(self.frames_data[f_num]) for f_num in self.frame_list}
        
        print(f"HDEPICFrameDataset initialised for {self.video_id} with {len(self.frame_list)} frames.")

    def _build_frame_registry(self):
        registry = {}
        for obj_id, obj_data in self.assoc_info.items():
            obj_name = obj_data['name']
            for track in obj_data.get('tracks', []):
                for frame_id in track.get('all_frames', []):
                    if frame_id not in self.frame_info:
                        continue
                        
                    f_info = self.frame_info[frame_id]
                    loc_3d = f_info.get('3d_location')
                    if loc_3d is None:
                        continue
                    
                    mask_filename = f"{frame_id}.png"
                    mask_path = None
                    if os.path.exists(os.path.join(self.unattached_dir, mask_filename)):
                        mask_path = os.path.join(self.unattached_dir, mask_filename)
                    elif os.path.exists(os.path.join(self.masks_dir, mask_filename)):
                        mask_path = os.path.join(self.masks_dir, mask_filename)
                        
                    if mask_path is None:
                        continue 
                    
                    frame_num = f_info['frame_number']
                    if frame_num not in registry:
                        registry[frame_num] = {} 
                    
                    bbox = f_info['bbox']
                    
                    if obj_id in registry[frame_num]:
                        existing_bbox = registry[frame_num][obj_id]['bbox']
                        merged_bbox = [
                            min(existing_bbox[0], bbox[0]),
                            min(existing_bbox[1], bbox[1]),
                            max(existing_bbox[2], bbox[2]),
                            max(existing_bbox[3], bbox[3])
                        ]
                        registry[frame_num][obj_id]['bbox'] = merged_bbox
                        
                        registry[frame_num][obj_id]['3d_locations'].append(loc_3d)
                        registry[frame_num][obj_id]['mask_paths'].append(mask_path)
                    else:
                        registry[frame_num][obj_id] = {
                            'obj_name': obj_name,
                            'bbox': bbox,
                            '3d_locations': [loc_3d],
                            'mask_paths': [mask_path] 
                        }
                        
        final_registry = {}
        for f_num, objs_dict in registry.items():
            objects_list = []
            for o_id, obj_data in objs_dict.items():
                avg_3d = np.mean(obj_data['3d_locations'], axis=0).tolist()
                objects_list.append({
                    'obj_id': o_id,
                    'obj_name': obj_data['obj_name'],
                    'bbox': obj_data['bbox'],
                    '3d_location': avg_3d,
                    'mask_paths': obj_data['mask_paths']
                })
            
            final_registry[f_num] = sorted(objects_list, key=lambda x: x['obj_id'])
            
        return final_registry

    def __getitem__(self, idx):
        frame_num = self.frame_list[idx]
        objects = self.frames_data[frame_num]
        frame = self.video_reader[frame_num].asnumpy()
        
        h, w = frame.shape[:2]
        obj_masks = []
        for obj in objects:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for m_path in obj['mask_paths']:
                raw_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                combined_mask |= (raw_mask > 0).astype(np.uint8)
            obj_masks.append(combined_mask)

        return {
            'video_id': self.video_id,
            'frame_number': frame_num,
            'frame': frame,
            'obj_ids': [obj['obj_id'] for obj in objects],
            'obj_names': [obj['obj_name'] for obj in objects],
            'obj_masks': np.array(obj_masks), 
            'bboxes': [list(map(int, obj['bbox'])) for obj in objects],
            '3d_locations': [obj['3d_location'] for obj in objects]
        }

    @staticmethod
    def process_item(data, transform_2d):
        if data is None: 
            return None
            
        frame_num = data['frame_number']
        frame = data['frame']
        obj_masks = data['obj_masks']
        bboxes = data['bboxes']
        
        frame_annotations = {
            "frame_name": f"frame_{frame_num:010d}", 
            "annotations": []
        }
        
        all_crops = []
        for i, bbox in enumerate(bboxes):
            obj_mask = obj_masks[i]

            frame_annotations["annotations"].append({
                "id": data['obj_ids'][i], 
                "name": data['obj_names'][i], 
                "bounding_box": list(map(int, bbox))
            })
            
            crop = create_object_crop(
                frame, bbox, input_mask=obj_mask,
                scale_frame_to_mask=False
            )
            
            all_crops.append(transform_2d(crop))
            
        stacked_crops = torch.stack(all_crops)

        return (
            frame_num, len(bboxes), stacked_crops, 
            data['3d_locations'], frame_annotations
        )

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch: 
            return None
        
        unzipped = list(zip(*batch))
        return {
            'frame_numbers': list(unzipped[0]),
            'num_objs_per_frame': list(unzipped[1]),
            'object_crops': torch.cat(unzipped[2]).pin_memory(),
            '3d_locations': list(unzipped[3]),
            'frame_annotations': list(unzipped[4])
        }

    def shutdown(self):
        if hasattr(self, 'video_reader'):
            del self.video_reader
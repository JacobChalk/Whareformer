import os
import cv2
import pandas as pd

from feature_extraction.datasets.base_dataset import BaseFeatureDataset

class IT3DEgoFrameDataset(BaseFeatureDataset):
    def __init__(self, video_id, cfg):
        super().__init__()
        self.video_id = video_id
        self.root = cfg.get('paths.it3dego_root')
        self.raw_root = os.path.join(self.root, "raw_videos", video_id)
        
        labels_path = os.path.join(self.root, "annotations", video_id, "labels.csv")
        self.label_map = pd.read_csv(labels_path, header=None)[0].to_dict()

        bbox_dir = os.path.join(self.root, "annotations", video_id, "2d_bbox_annot")
        self.frames_data = self._build_frame_registry(bbox_dir)
        
        self.frame_list = sorted(list(self.frames_data.keys()), key=lambda x: float(x))
        self.obj_counts = {ts: len(self.frames_data[ts]) for ts in self.frame_list}
        
        pv_dir = os.path.join(self.raw_root, "pv")
        all_ts = [f.split('.')[0] for f in os.listdir(pv_dir) if f.endswith('.png')]
        all_ts = sorted(all_ts, key=lambda x: float(x))
        
        start_idx = all_ts.index(self.frame_list[0])
        self.ts_to_frame_name = {ts: f"frame_{(all_ts.index(ts) - start_idx) + 1:010d}" for ts in self.frame_list}

                        
        print(f"IT3DEgoFrameDataset initialised for {video_id} with {len(self.frame_list)} timestamps.")

    def _build_frame_registry(self, bbox_dir):
        registry = {}
        for filename in os.listdir(bbox_dir):
            if not filename.endswith(".txt"): 
                continue
            obj_idx = int(filename.split(".")[0])
            obj_name = self.label_map.get(obj_idx, f"unknown_{obj_idx}")
            
            with open(os.path.join(bbox_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: 
                        continue
                    ts, x, y, w, h = parts
                    bbox = [float(x), float(y), float(x) + float(w), float(y) + float(h)]
                    
                    if ts not in registry: 
                        registry[ts] = []
                    registry[ts].append({'obj_id': str(obj_idx), 'obj_name': obj_name, 'bbox': bbox})
        return registry

    def __getitem__(self, idx):
        ts = self.frame_list[idx]
        objects = self.frames_data[ts]
        
        frame_path = os.path.join(self.raw_root, "pv", f"{ts}.png")
        depth_path = os.path.join(self.raw_root, "depth_ahat", f"{ts}.png")
        
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        return {
            'timestamp': ts,
            'frame_name': self.ts_to_frame_name[ts],
            'frame': frame,
            'depth_map': depth_map,
            'bboxes': [obj['bbox'] for obj in objects],
            'obj_ids': [obj['obj_id'] for obj in objects],
            'obj_names': [obj['obj_name'] for obj in objects]
        }

    @staticmethod
    def process_item(data, transform_2d=None, **kwargs):
        if data is None: 
            return None
        return (
            data['timestamp'], data['frame_name'], len(data['bboxes']), data['bboxes'], 
            data['obj_ids'], data['obj_names'], data['frame'], data['depth_map']
        )

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: 
            return None
        
        unzipped = list(zip(*batch))
        return {
            'timestamps': list(unzipped[0]),
            'frame_names': list(unzipped[1]),
            'num_objs_per_frame': list(unzipped[2]),
            'bboxes': list(unzipped[3]),
            'obj_ids': list(unzipped[4]),
            'obj_names': list(unzipped[5]),
            'frames': list(unzipped[6]),
            'depth_maps': list(unzipped[7])
        }
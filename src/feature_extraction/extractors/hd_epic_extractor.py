import time
import torch
import numpy as np
from functools import partial

import feature_extraction.utils.models as models
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor
from feature_extraction.datasets.hd_epic_dataset import HDEPICFrameDataset
from feature_extraction.utils.loaders import ThreadPoolDataLoader, ObjectCountBatchSampler

class HDEPICFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, video_id, cfg):
        super().__init__(video_id, cfg)
        
        print(f"Loading 2D model for {video_id}...")
        self.model_2d, transform_2d, self.forward_fn_2d = models.load_2d_model(
            cfg.get('feature_extraction.model_2d'), self.device
        )

        frame_dataset = HDEPICFrameDataset(video_id, cfg)
        batch_sampler = ObjectCountBatchSampler(
            frame_dataset, 
            target_object_count=cfg.get('feature_extraction.target_batch_size', 128)
        )

        worker_fn = partial(
            frame_dataset.process_item, 
            transform_2d=transform_2d
        )

        self.data_loader = ThreadPoolDataLoader(
            dataset=frame_dataset,
            batch_sampler=batch_sampler,
            collate_fn=frame_dataset.collate_fn,
            worker_fn=worker_fn,
            num_workers=cfg.get('feature_extraction.num_workers', 8)
        )
        
        self.object_annotations = {"video_annotations": []}
        print(f"Initialisation complete for {video_id}.")

    def _process_batch(self, batch):
        if batch is None: 
            return None
            
        timings = {}
        t_start = time.perf_counter()

        for i, frame_num in enumerate(batch['frame_numbers']):
            self.object_annotations["video_annotations"].append(batch['frame_annotations'][i])
            self.save_dict_3D[f"frame_{frame_num:010d}"] = np.array(batch['3d_locations'][i], dtype=np.float32)

        object_crops = batch['object_crops'].to(self.device, non_blocking=True)
        with torch.no_grad():
            appearance_features = self.forward_fn_2d(self.model_2d, object_crops).cpu().numpy()
            
        current_idx = 0
        for frame_num, num_objs in zip(batch['frame_numbers'], batch['num_objs_per_frame']):
            self.save_dict_2D[f"frame_{frame_num:010d}"] = appearance_features[current_idx : current_idx + num_objs].astype(np.float32)
            current_idx += num_objs
                
        timings['t_2d'] = time.perf_counter() - t_start
        return timings

    def shutdown(self):
        self.data_loader.dataset.shutdown()
        del self.model_2d, self.data_loader
        super().shutdown()
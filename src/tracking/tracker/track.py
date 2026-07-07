import numpy as np

from .memory.denstream import FeatureDenStream
from .memory.buffer import FeatureBuffer

class Track:
    def __init__(self, track_id, frame_metadata, detection=None,
                 app_memory=100, loc_memory=100, 
                 app_aggregation='nearest_neighbour', loc_aggregation='nearest_neighbour',
                 max_radius=0.25, mu=10.0, decay_rate=1e-3):
        
        self.track_id       = track_id
        self.time_since_update = 0
        self.initial_object = detection['obj_name']
        self.metadata_keys  = ["frame", "detection_idx", "bbox", "obj_name"]

        timestamp = frame_metadata['timestamp']
        self.last_seen_time = timestamp
        self.creation_time  = frame_metadata["frame_name"]

        self.app_aggregation = app_aggregation
        self.loc_aggregation = loc_aggregation

        self.last_loc = detection['loc']

        def _init_repr(agg_type, memory_limit):
            if agg_type == 'clusters':
                return FeatureDenStream(
                    timestamp, max_radius=float(max_radius), 
                    mu=float(mu), decay_rate=float(decay_rate)
                )
            return FeatureBuffer(chunk_size=1000, max_memory=memory_limit)

        self.app_repr = _init_repr(self.app_aggregation, app_memory)
        self.loc_repr = _init_repr(self.loc_aggregation, loc_memory)

        # Do initial update
        self._update_repr(self.app_repr, self.app_aggregation, detection['app'], timestamp)
        self._update_repr(self.loc_repr, self.loc_aggregation, self.last_loc, timestamp)

        self.history = [{k: detection[k] for k in self.metadata_keys}]

    def _update_repr(self, model, agg_type, feature, timestamp):
        """Helper to handle the different update signatures."""
        if agg_type == 'clusters':
            model.update(feature, timestamp)
        else:
            model.update(feature)

    def increase_age(self, timestamp):
        self.time_since_update = timestamp - self.last_seen_time

    def update(self, detection, frame_metadata):
        timestamp = frame_metadata['timestamp']
        self.time_since_update = 0
        self.last_seen_time = timestamp

        self.last_loc = detection['loc']

        self._update_repr(self.app_repr, self.app_aggregation, detection['app'], timestamp)
        self._update_repr(self.loc_repr, self.loc_aggregation, self.last_loc, timestamp)
        
        self.history.append({k: detection[k] for k in self.metadata_keys})

    def _get_repr(self, representaiton, agg_type):
        """Helper for fetching representations."""
        if agg_type == 'clusters':
            return representaiton.get_clusters()
            
        buffer_array = representaiton.get_array()
        if agg_type == 'mean':
            return np.mean(buffer_array, axis=0, keepdims=True)
                
        return buffer_array

    def get_appearance_representation(self):
        return self._get_repr(self.app_repr, self.app_aggregation)

    def get_location_representation(self):
        return self._get_repr(self.loc_repr, self.loc_aggregation)

    def get_observations(self):
        return self.get_appearance_representation(), self.get_location_representation()

    def get_recent_assignment(self):
        h = self.history[-1]
        return (h['obj_name'], h['bbox'], self.last_loc)

    def __repr__(self):
        return f"Track ID: {self.track_id} {self.initial_object} {[d['obj_name'] for d in self.history]}\n"
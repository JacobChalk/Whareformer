from abc import ABC, abstractmethod
from .track import Track
from .matcher.matcher_registry import build_matcher

class BaseTracker(ABC):
    def __init__(self, tracking_params, fps=60):
        self.fps = fps
        self.tracks = []
        self.last_track_id = 0
        
        matcher_params = tracking_params.get('matcher_params', {})
        
        def parse_memory(representation):
            if representation == 'most_recent': return 1
            elif representation == 'one_second_window': return self.fps
            elif representation == 'buffer': return matcher_params.get('memory_buffer_size', 100)
            elif representation in ['denstream', 'infinite']: return None
            else: raise ValueError(f"Representation Strategy: {representation} is not defined!")

        self.app_memory = parse_memory(matcher_params.get('app_representation', 'buffer'))
        self.loc_memory = parse_memory(matcher_params.get('loc_representation', 'buffer'))
        
        self.app_aggregation = matcher_params.get('app_aggregation', 'nearest_neighbour')
        self.loc_aggregation = matcher_params.get('loc_aggregation', 'nearest_neighbour')
            
        self.denstream_params = matcher_params.get(
            'denstream_params', {'max_radius': 0.25, 'mu': 10.0, 'decay_rate': 1e-3}
        )

    @abstractmethod
    def update(self, detections, frame_metadata):
        pass

    def clear(self):        
        self.tracks.clear()
        self.last_track_id = 0
        
    def _create_track(self, detection, track_id, frame_metadata):
        return Track(
            track_id, frame_metadata,
            detection=detection.detection_data,
            app_memory=self.app_memory,
            loc_memory=self.loc_memory,
            app_aggregation=self.app_aggregation,
            loc_aggregation=self.loc_aggregation,
            max_radius=self.denstream_params['max_radius'],
            mu=self.denstream_params['mu'],
            decay_rate=self.denstream_params['decay_rate']
        )


class StandardTracker(BaseTracker):
    def __init__(self, tracking_params, fps=60):
        super().__init__(tracking_params, fps)
        self.matcher_type = tracking_params.get('matcher_type', 'lmk')
        
        matcher_params = tracking_params.get('matcher_params', {})
        matcher_params.update({
            'model': tracking_params.get('model', {}), 
            'device_id': tracking_params.get('device_id', None), 
            'fps': fps,
            'app_dim': tracking_params.get('app_dim', 256) 
        })
        self.matcher = build_matcher(self.matcher_type, matcher_params)

    def update(self, detections, frame_metadata):
        # Age existing tracks
        for track in self.tracks:
            track.increase_age(frame_metadata['timestamp'])

        # Get assignments from matcher
        matches, unmatched_tracks, new_tracks, training_data = \
            self.matcher.match(self.tracks, detections, frame_metadata)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx].detection_data, frame_metadata)

        # Spawn new tracks
        for detection_idx in new_tracks:
            new_track = self._create_track(detections[detection_idx], self.last_track_id, frame_metadata)
            self.tracks.append(new_track)
            self.last_track_id += 1

        return training_data
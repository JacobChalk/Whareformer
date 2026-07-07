import numpy as np
import torch

from .matcher_registry import register_matcher
from utils.distance import pdist_dot, pdist_euclidean
from models import get_model 

class TrackMatcher(object):
    def __init__(self, matcher_type, matching_params):
        self.matcher_type = matcher_type
        self.app_dim = matching_params.get('app_dim', 256)

        device_id = matching_params.get('device_id', None)
        if device_id is not None:
            torch.cuda.set_device(device_id)
            self.device = f'cuda:{device_id}'
        else:
            self.device = 'cpu'

    def _get_representations(self, tracks, detections):
        det_app_embs = np.array([d.detection_data['app'] for d in detections])
        det_loc_embs = np.array([d.detection_data['loc'] for d in detections])
        detection_features = list(zip(det_app_embs, det_loc_embs))

        track_app_embs = [t.get_appearance_representation() for t in tracks]
        track_loc_embs = [t.get_location_representation() for t in tracks]
        
        track_features = list(zip(track_app_embs, track_loc_embs))

        return track_features, detection_features

@register_matcher("oracle")
class OracleMatcher(TrackMatcher):
    def __init__(self, matcher_type, matching_params):
        super().__init__(matcher_type, matching_params)

        self.app_dist_fn = pdist_dot
        self.loc_dist_fn = pdist_euclidean

    def _construct_feature_matrices(self, track_reprs, det_reprs):
        det_app_features = np.array([d[0] for d in det_reprs], dtype=np.float32)
        det_loc_features = np.array([d[1] for d in det_reprs], dtype=np.float32)

        track_apps = [np.atleast_2d(t[0]).astype(np.float32, copy=False) for t in track_reprs]
        track_locs = [np.atleast_2d(t[1]).astype(np.float32, copy=False) for t in track_reprs]

        app_lengths = [len(a) for a in track_apps]
        loc_lengths = [len(l) for l in track_locs]

        app_starts = np.r_[0, np.cumsum(app_lengths[:-1])]
        loc_starts = np.r_[0, np.cumsum(loc_lengths[:-1])]

        all_track_apps = np.concatenate(track_apps, axis=0)
        all_track_locs = np.concatenate(track_locs, axis=0)

        app_dists_flat = self.app_dist_fn(det_app_features, all_track_apps)
        loc_dists_flat = self.loc_dist_fn(det_loc_features, all_track_locs)

        app_chunks = np.split(app_dists_flat, app_starts[1:], axis=1)
        loc_chunks = np.split(loc_dists_flat, loc_starts[1:], axis=1)

        app_local_mins = [np.argmin(chunk, axis=1) for chunk in app_chunks]
        loc_local_mins = [np.argmin(chunk, axis=1) for chunk in loc_chunks]

        app_nn_global = (np.column_stack(app_local_mins) + app_starts).astype(np.intp)
        loc_nn_global = (np.column_stack(loc_local_mins) + loc_starts).astype(np.intp)

        track_app_features = all_track_apps[app_nn_global]
        track_loc_features = all_track_locs[loc_nn_global]

        return det_app_features, det_loc_features, track_app_features, track_loc_features

    def match(self, tracks, detections, frame_metadata):
        if len(tracks) == 0:
            return [], [], list(range(len(detections))), None

        track_reprs, det_reprs = self._get_representations(tracks, detections)
        features = self._construct_feature_matrices(track_reprs, det_reprs)

        track_name_to_idx = {track.initial_object: idx for idx, track in enumerate(tracks)}

        matches = []
        matched_track_indices = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(detections):
            obj_name = det.detection_data['obj_name']
            track_idx = track_name_to_idx.get(obj_name)
            
            if track_idx is not None:
                matches.append((track_idx, det_idx))
                matched_track_indices.add(track_idx)
                matched_det_indices.add(det_idx)

        unmatched_tracks = [t for t in range(len(tracks)) if t not in matched_track_indices]
        new_tracks = [d for d in range(len(detections)) if d not in matched_det_indices]

        training_data = (features, matches, new_tracks)

        return matches, unmatched_tracks, new_tracks, training_data

@register_matcher("whareformer")    
class WhareformerMatcher(TrackMatcher):
    def __init__(self, matcher_type, matching_params):
        super().__init__(matcher_type, matching_params)
        
        self.app_dist_fn = pdist_dot
        self.loc_dist_fn = pdist_euclidean

        model_path = matching_params.get('model_path', None)

        if model_path is None:
            raise ValueError("Model path must be provided.")

        self.model_config = matching_params.get('model', {})
        architecture = self.model_config.get('architecture', {})

        try:
            self.model = get_model('whareformer', **architecture).to(self.device)
            state_dict = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"{e}\nFailed to load model from {model_path}. Please check the path and model parameters and try again.")

    def _construct_model_inputs(self, track_reprs, det_reprs):
        num_tracks = len(track_reprs)
        num_dets = len(det_reprs)

        det_apps = np.array([d[0] for d in det_reprs], dtype=np.float32)
        det_locs = np.array([d[1] for d in det_reprs], dtype=np.float32)

        track_apps = [np.atleast_2d(t[0]).astype(np.float32, copy=False) for t in track_reprs]
        track_locs = [np.atleast_2d(t[1]).astype(np.float32, copy=False) for t in track_reprs]

        app_lengths = [len(a) for a in track_apps]
        loc_lengths = [len(l) for l in track_locs]

        app_starts = np.r_[0, np.cumsum(app_lengths[:-1])]
        loc_starts = np.r_[0, np.cumsum(loc_lengths[:-1])]

        all_track_apps = np.concatenate(track_apps, axis=0)
        all_track_locs = np.concatenate(track_locs, axis=0)

        app_dists_flat = self.app_dist_fn(det_apps, all_track_apps)
        loc_dists_flat = self.loc_dist_fn(det_locs, all_track_locs)

        app_chunks = np.split(app_dists_flat, app_starts[1:], axis=1)
        loc_chunks = np.split(loc_dists_flat, loc_starts[1:], axis=1)

        app_local_mins = [np.argmin(chunk, axis=1) for chunk in app_chunks]
        loc_local_mins = [np.argmin(chunk, axis=1) for chunk in loc_chunks]

        app_nn_global = (np.column_stack(app_local_mins) + app_starts).astype(np.intp)
        loc_nn_global = (np.column_stack(loc_local_mins) + loc_starts).astype(np.intp)

        track_app_features = all_track_apps[app_nn_global]
        track_loc_features = all_track_locs[loc_nn_global]

        # --- CONSTRUCT MODEL INPUTS ---
        app_dim = det_apps.shape[1]
        loc_dim = det_locs.shape[1]
        total_dim = app_dim + loc_dim
        
        model_inputs = np.zeros((num_dets, num_tracks + 1, total_dim), dtype=np.float32)

        model_inputs[:, 0, :app_dim] = det_apps
        model_inputs[:, 0, app_dim:] = det_locs

        model_inputs[:, 1:, :app_dim] = track_app_features
        model_inputs[:, 1:, app_dim:] = track_loc_features

        return model_inputs

    def match(self, tracks, detections, frame_metadata):
        if len(tracks) == 0:
            return [], [], list(range(len(detections))), None

        track_reprs, det_reprs = self._get_representations(tracks, detections)
        model_inputs = self._construct_model_inputs(track_reprs, det_reprs)

        with torch.no_grad():
            input_tensor = torch.from_numpy(model_inputs).to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        matches, unmatched_tracks, new_tracks = self._greedy_assignment(probs)
        self._validate_outputs(matches, unmatched_tracks, new_tracks, len(tracks), len(detections))

        return matches, unmatched_tracks, new_tracks, None

    def _greedy_assignment(self, assignment_probs):
        num_dets, num_cols = assignment_probs.shape
        num_tracks = num_cols - 1

        matches = []
        new_tracks = []
        assigned_tracks = set()

        probs = assignment_probs.astype(np.float32, copy=True)

        for _ in range(num_dets):
            flat_idx = np.argmax(probs)
            r, c = np.unravel_index(flat_idx, probs.shape)

            if probs[r, c] == -np.inf:
                break

            probs[r, :] = -np.inf

            if c == 0:
                new_tracks.append(int(r))
            else:
                track_idx = c - 1
                matches.append((track_idx, int(r)))
                assigned_tracks.add(track_idx)
                probs[:, c] = -np.inf

        unmatched_tracks = [t for t in range(num_tracks) if t not in assigned_tracks]
        return matches, unmatched_tracks, new_tracks

    def _validate_outputs(self, matches, unmatched_tracks, unmatched_dets, num_tracks, num_dets):
        assert len(matches) + len(unmatched_dets) == num_dets, \
            f'Detection mismatch! {len(matches)}+{len(unmatched_dets)} != {num_dets}'
        assert len(matches) + len(unmatched_tracks) == num_tracks, \
            f'Track mismatch! {len(matches)}+{len(unmatched_tracks)} != {num_tracks}'
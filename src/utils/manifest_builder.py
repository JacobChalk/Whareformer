import numpy as np
import hashlib

def hash_numpy_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def build_feature_manifest_entry(
    matches, 
    unmatched_detections, 
    features, 
    video_id, 
    timestamp, 
    txn, 
    feature_cache: dict
):
    if features is None:
        return []
    
    det_app_feats, det_loc_feats, track_app_feats, track_loc_feats = features
    D, T, _ = track_app_feats.shape
    
    if D == 0:
        return []

    manifest_entries = []

    def get_or_set_feature(feature_array: np.ndarray, key_prefix: str) -> str:
        feature_hash = hash_numpy_array(feature_array)
        
        if feature_hash in feature_cache:
            return feature_cache[feature_hash]
        else:
            lmdb_key = f"{key_prefix}/{feature_hash}".encode('utf-8')
            txn.put(lmdb_key, feature_array.tobytes())
            
            str_key = lmdb_key.decode('utf-8')
            feature_cache[feature_hash] = str_key
            return str_key

    for d_idx in range(D):
        det_app = det_app_feats[d_idx]
        det_app_key = get_or_set_feature(det_app, f"{video_id}/{timestamp}/feature/app")
        
        det_loc = det_loc_feats[d_idx]
        det_loc_key = get_or_set_feature(det_loc, f"{video_id}/{timestamp}/feature/loc")

        track_app_keys = []
        track_loc_keys = []

        for t_idx in range(T):
            track_app = track_app_feats[d_idx, t_idx]
            track_app_key = get_or_set_feature(track_app, f"{video_id}/{timestamp}/feature/app")
            track_app_keys.append(track_app_key)
            
            track_loc = track_loc_feats[d_idx, t_idx]
            track_loc_key = get_or_set_feature(track_loc, f"{video_id}/{timestamp}/feature/loc")
            track_loc_keys.append(track_loc_key)

        label = np.zeros(T + 1, dtype=np.int8)
        is_matched = False
        for t_match, d_match in matches:
            if d_match == d_idx:
                label[t_match + 1] = 1
                is_matched = True
                break
        
        if not is_matched and d_idx in unmatched_detections:
            label[0] = 1

        manifest_entries.append({
            'det_app_key': det_app_key,
            'det_loc_key': det_loc_key,
            'track_app_keys': track_app_keys,
            'track_loc_keys': track_loc_keys,
            'label': label
        })
        
    return manifest_entries
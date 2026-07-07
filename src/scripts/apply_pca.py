import os
import json
import pickle
import argparse
import numpy as np
import random

from sklearn.decomposition import PCA
from tqdm import tqdm

def generate_seed():
    return int.from_bytes(os.urandom(8), byteorder="big")

def set_seed(seed: int = 0):
    if seed < 0:
        seed = generate_seed()
    seed = seed % (2**32)
    print(f"Using Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def l2_normalise(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

def load_video_feature(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_video_feature(out_path, frame_dict):
    with open(out_path, 'wb') as f:
        pickle.dump(frame_dict, f)


def fit_pca_on_train_videos(output_dir, train_ids, output_dim, seed, feat_type):
    object_features = {}
    
    for vid in tqdm(train_ids, desc="[PCA] Collecting train features"):
        pkl_path = os.path.join(output_dir, vid, f"2D_feat_{feat_type}.pkl")
        json_path = os.path.join(output_dir, vid, "object_annotations.json")
        
        if not os.path.exists(pkl_path) or not os.path.exists(json_path):
            continue
            
        frame_dict = load_video_feature(pkl_path)
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        frame_to_ann = {item['frame_name']: item['annotations'] for item in json_data['video_annotations']}

        for frame_name, feats in frame_dict.items():
            anns = frame_to_ann.get(frame_name, [])
            if len(anns) != feats.shape[0]:
                continue
                
            normed_feats = l2_normalise(feats)
        
            for i, ann in enumerate(anns):
                obj_id = f"{vid}_{ann['name']}"
                if obj_id not in object_features:
                    object_features[obj_id] = []
                object_features[obj_id].append(normed_feats[i])

    if not object_features:
        raise ValueError("No features were collected.")

    counts = [len(feats) for feats in object_features.values()]
    median_cap = int(np.median(counts))
    print(f"[PCA] Object counts - Min: {min(counts)}, Max: {max(counts)}, Median Cap: {median_cap}")

    balanced_feats = []
    for obj_id, feats in object_features.items():
        feats_array = np.vstack(feats)
        num_feats = feats_array.shape[0]
        
        if num_feats > median_cap:
            indices = np.linspace(0, num_feats - 1, median_cap, dtype=int)
            feats_array = feats_array[indices]
            
        balanced_feats.append(feats_array)

    balanced_feats = np.vstack(balanced_feats)
    print(f"[PCA] Fitting on {balanced_feats.shape[0]} balanced observations ({len(object_features)} unique objects)")

    pca = PCA(n_components=output_dim, whiten=True, svd_solver='full', random_state=seed)
    pca.fit(balanced_feats)
    
    return pca

def process_and_save_all_videos(output_dir, video_ids, pca, output_dim, feat_type):
    for vid in tqdm(video_ids, desc="[Apply PCA]"):
        pkl_path = os.path.join(output_dir, vid, f"2D_feat_{feat_type}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[WARN] Missing file: {pkl_path}")
            continue

        frame_dict = load_video_feature(pkl_path)
        reduced_dict = {}
        for frame_id, feats in frame_dict.items():
            feats = l2_normalise(feats)                 # Pre-PCA normalization
            feats_reduced = pca.transform(feats)        # Apply PCA
            feats_reduced = l2_normalise(feats_reduced) # Post-PCA normalization
            reduced_dict[frame_id] = feats_reduced

        save_path = os.path.join(output_dir, vid, f"2D_feat_{feat_type}_PCA_{output_dim}d.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video_feature(save_path, reduced_dict)

def main(args):
    set_seed(args.seed)
    all_videos = [v for v in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, v))]
    
    if args.pca_path is not None:
        pca = pickle.load(open(args.pca_path, 'rb'))['pca']
    else:
        from utils import Config
        cfg = Config(args.config_path)
        train_videos = cfg.get('videos').get('train', [])
        train_videos = [vid for vid in train_videos if vid in all_videos]
        
        if not train_videos:
            raise ValueError("No Training videos found!")
        print(f"Fitting PCA on {len(train_videos)} train videos, applying to {len(all_videos)} total videos")

        pca = fit_pca_on_train_videos(args.output_dir, train_videos, args.output_dim, args.seed, args.feat_type)
        
        os.makedirs(args.output_dir, exist_ok=True)
        pca_path = os.path.join(args.output_dir, f'PCA_model_{args.feat_type}_{args.output_dim}d.pkl')
        pickle.dump({'pca': pca}, open(pca_path, 'wb'))
        print(f"Saved PCA model to {pca_path}")

    process_and_save_all_videos(args.output_dir, all_videos, pca, args.output_dim, args.feat_type)
    print(f"All videos processed and saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to load/save {video_id}/2D_feat_<feat_type>.pkl files")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to config containing training/test split")
    parser.add_argument("--pca_path", type=str, default=None, 
                        help="Pretrained PCA Path")
    parser.add_argument("--output_dim", type=int, default=256,
                        help="Dimensionality after PCA")
    parser.add_argument("--feat_type", choices=['bbox', 'sam_masked', 'masked'], default='masked',
                        help="What type of features to learn")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random Seed")

    args = parser.parse_args()
    main(args)
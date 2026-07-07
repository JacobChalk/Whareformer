import os
import pickle
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_idf1_worker(results):
    frame_updates = []
    all_frames = [int(f.split('_')[1]) if isinstance(f, str) else f for f in results.keys()]
    all_frames = np.sort(np.array(all_frames))
    
    if len(all_frames) == 0:
        return frame_updates
    
    gt_id_map = {}       
    
    for t in all_frames:
        t_key = f'frame_{t:010d}'
        if len(results[t_key]['track_obj_names']) == 0:
            continue
        
        gt_ids, pred_ids = [], []
        gt_boxes, pred_boxes = [], []

        for obj_n, obj in enumerate(results[t_key]['track_obj_names']):
            if obj not in gt_id_map:
                gt_id_map[obj] = len(gt_id_map)
                
            if gt_id_map[obj] in gt_ids:
                continue
                
            bbox = list(map(int, results[t_key]['track_bboxes'][obj_n]))

            gt_ids.append(gt_id_map[obj])
            gt_boxes.append(bbox)

            pred_ids.append(results[t_key]['track_ids'][obj_n])
            pred_boxes.append(bbox)

        if len(gt_ids) != len(set(gt_ids)):
            raise ValueError(f"Duplicate GT IDs in frame {t_key}")
        if len(pred_ids) != len(set(pred_ids)):
            raise ValueError(f"Duplicate Pred IDs in frame {t_key}")
            
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        frame_updates.append((gt_ids, pred_ids, dist))
        
    return frame_updates

def evaluate_video(video_dir, video_name):
    csv_path = os.path.join(video_dir, 'results.csv')
    pkl_path = os.path.join(video_dir, 'tracking_outputs.pkl')

    correct_array = np.zeros(145)
    total_array = np.zeros(145)
    frame_updates = []

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=',', nrows=145)
        
        correct_col = next((c for c in df.columns if c.endswith('_correct')), None)
        total_col = next((c for c in df.columns if c.endswith('_total')), None)
        
        if correct_col and total_col:
            pcl_col = correct_col.replace('_correct', '_pcl')
            has_timescale = (df[pcl_col] != -1).astype(int).values
            
            correct_array = df[correct_col].values * has_timescale
            total_array = df[total_col].values * has_timescale

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            tracking_results = pickle.load(f)
        frame_updates = process_idf1_worker(tracking_results)

    return video_name, correct_array, total_array, frame_updates

def evaluate_directory(results_dir):
    if not os.path.exists(results_dir):
        print("Directory does not exist.")
        return

    videos = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    videos = sorted([v for v in videos if os.path.exists(os.path.join(results_dir, v, 'results.csv'))])

    if not videos:
        print("No valid video results found.")
        return

    accumulators = []
    valid_idf1_videos = []
    
    agg_correct = np.zeros(145)
    agg_total = np.zeros(145)
    
    num_workers = min(32, os.cpu_count() or 8)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_video, os.path.join(results_dir, video), video): video 
            for video in videos
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            try:
                video_name, correct_arr, total_arr, frame_updates = future.result()
                
                agg_correct += correct_arr
                agg_total += total_arr
                
                if frame_updates:
                    acc = mm.MOTAccumulator(auto_id=True)
                    for gt_ids, pred_ids, dist in frame_updates:
                        acc.update(gt_ids, pred_ids, dist)
                        
                    accumulators.append(acc)
                    valid_idf1_videos.append(video_name)
                    
            except Exception as e:
                print(f"Error processing video {futures[future]}: {e}")

    print("--- Final Results ---")
    
    # Compute mPCL
    safe_total = np.where(agg_total == 0, 1, agg_total)
    mpcl = 100.0 * np.mean(agg_correct / safe_total)
    print(f"mPCL: {mpcl:.1f}%")

    # Compute IDF1
    if accumulators:
        mh = mm.metrics.create()
        summary = mh.compute_many(accumulators, metrics=['idf1'], names=valid_idf1_videos, generate_overall=True)
        overall_idf1 = summary.loc['OVERALL']['idf1'] * 100
        print(f"IDF1: {overall_idf1:.1f}%")
    else:
        print("IDF1: N/A (No tracking_outputs.pkl files found)")
        
    print("-" * 21)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tracking Performance (IDF1 & mPCL)")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True, 
        help="Path to the parent directory containing video subfolders"
    )
    args = parser.parse_args()
    
    evaluate_directory(args.results_dir)
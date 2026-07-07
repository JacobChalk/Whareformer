import math
import bisect

from collections import defaultdict

def extract_frame_number(frame_name):
    return int(frame_name.split('_')[1])

def euclidean_distance(loc1, loc2):
    return math.dist(loc1, loc2)

def get_tracking_data(results):
    obj_sets = {}
    obj_track_ids = {} 
    obj_locs = {}
    track_locs = {}

    for frame_name in sorted(list(results.keys())):
        frame_num = extract_frame_number(frame_name)
        obj_set = set()

        for o, obj_name in enumerate(results[frame_name]['track_obj_names']):
            obj_set.add(obj_name)
            track_id = results[frame_name]['track_ids'][o]
            location = results[frame_name]['track_3d_locs'][o]

            if obj_name not in obj_track_ids:
                obj_track_ids[obj_name] = ([], [])
            if obj_name not in obj_locs:
                obj_locs[obj_name] = ([], [])
            if track_id not in track_locs:
                track_locs[track_id] = ([], [])

            obj_track_ids[obj_name][0].append(frame_num)
            obj_track_ids[obj_name][1].append(track_id)

            obj_locs[obj_name][0].append(frame_num)
            obj_locs[obj_name][1].append(location)
            
            track_locs[track_id][0].append(frame_num)
            track_locs[track_id][1].append(location)

        obj_sets[frame_name] = obj_set
    
    track_lifespans = defaultdict(list)
    
    for _, (frames, t_ids) in obj_track_ids.items():
        current_track = t_ids[0]
        current_start = -float('inf') # The first track gets the retroactive past
        
        for i in range(1, len(frames)):
            if t_ids[i] != current_track:
                track_lifespans[current_track].append((current_start, frames[i] - 1))
                
                current_track = t_ids[i]
                current_start = frames[i]
                
        track_lifespans[current_track].append((current_start, float('inf')))

    return obj_sets, obj_track_ids, obj_locs, track_locs, track_lifespans

def get_last_known_state(history_tuple, target_frame_num):
    """
    Binary search to find the last known location of an object/track closest to the target frame.
    Assumes objects remain at rest (forward-fills last known state).
    """ 
    seen_frame_nums, values = history_tuple
        
    idx = max(0, bisect.bisect_right(seen_frame_nums, target_frame_num) - 1)
        
    return values[idx]

def get_timescale_accuracy(
    key_frames,
    timescale,
    dist_thresh,
    obj_track_ids,
    obj_locs,
    track_locs,
    track_lifespans,
    final_frame,
    dense_eval=True
):
    correct_count, total_count = 0, 0
    max_frame_num = extract_frame_number(final_frame)

    for key_frame, objects in key_frames.items():
        key_frame_num = extract_frame_number(key_frame)
        
        for obj in objects:
            key_frame_obj_track_id = get_last_known_state(obj_track_ids[obj], key_frame_num)

            timescale_frames = [timescale, -timescale] if dense_eval else [timescale]
            
            for delta in timescale_frames:
                target_frame_num = key_frame_num + delta

                if target_frame_num < 0 or target_frame_num > max_frame_num:
                    continue
                
                track_alive = False
                for start, end in track_lifespans[key_frame_obj_track_id]:
                    if start <= target_frame_num <= end:
                        track_alive = True
                        break
                
                if not track_alive:
                    total_count += 1
                    continue

                obj_loc = get_last_known_state(obj_locs[obj], target_frame_num)
                track_loc = get_last_known_state(track_locs[key_frame_obj_track_id], target_frame_num)

                dist = euclidean_distance(obj_loc, track_loc)
                if dist <= dist_thresh:
                    correct_count += 1
                        
                total_count += 1

    percentage = (correct_count / total_count) * 100 if total_count > 0 else -1
    return percentage, correct_count, total_count

def get_dense_key_frames(obj_names, min_count):
    all_object_sets = defaultdict(list)
    for frame_idx, objs in obj_names.items():
        obj_set = tuple(sorted(objs))
        all_object_sets[obj_set].append(frame_idx)

    valid_sets = {}
    for threshold in range(min_count, 0, -1):
        valid_sets = {obj_set: frames for obj_set, frames in all_object_sets.items() if len(obj_set) >= threshold}
        if valid_sets:
            break

    key_frames = defaultdict(set)
    for obj_set, frames in valid_sets.items():
        median_idx = len(frames) // 2
        key_frames[frames[median_idx]] = obj_set

    return dict(key_frames)

def get_sparse_key_frames(obj_names, fps=30, stride_time=12*60):
    object_frames = defaultdict(list)
    for frame_name, objs in obj_names.items():
        frame_num = extract_frame_number(frame_name)
        for obj in objs:
            object_frames[obj].append(frame_num)

    stride = int(stride_time * fps)
    key_frames = defaultdict(set)

    for obj, frame_nums in object_frames.items():
        frame_nums.sort()
        current_frame_num = frame_nums[0]
        key_frames[f"frame_{current_frame_num:010d}"].add(obj)
        
        while True:
            target_frame_num = current_frame_num + stride
            
            idx = bisect.bisect_left(frame_nums, target_frame_num)
            
            if idx == len(frame_nums):
                break
                
            current_frame_num = frame_nums[idx]
            key_frames[f"frame_{current_frame_num:010d}"].add(obj)

    return dict(key_frames)
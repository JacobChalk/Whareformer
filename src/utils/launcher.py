import os
import sys
import multiprocessing

from multiprocessing import Manager
from itertools import zip_longest

worker_cfg = None

def init_worker_config(config_file_path: str):
    global worker_cfg
    print(f"[Worker {os.getpid()}] Initialising Config from: {config_file_path}")
    from utils import Config
    worker_cfg = Config(config_file_path)
    print(f"[Worker {os.getpid()}] Config initialised.")

def pool_worker_task(task_fn, gpu_queue, use_gpus, video_id):
    global worker_cfg
    if worker_cfg is None:
        print(f"[Worker {os.getpid()}] ERROR: Worker config is None. This should not happen.")
        return
    
    gpu_id = None    
    try:
        if use_gpus:
            gpu_id = gpu_queue.get()
            print(f"[Worker {os.getpid()}] Acquired GPU {gpu_id} for: {video_id}")
        else:
            print(f"[Worker {os.getpid()}] Starting CPU task for: {video_id}")

        task_fn(worker_cfg, video_id=video_id, gpu_id=gpu_id)

        print(f"[Worker {os.getpid()}] Finished task for: {video_id}")

    except Exception as e:
        print(f"[Worker {os.getpid()}] ERROR processing {video_id} on GPU {gpu_id}: {e}", file=sys.stderr)
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Worker {os.getpid()}] Released GPU {gpu_id}")

def get_tracking_split_videos(cfg):
    split = cfg.get('tracking').get('split', 'train')
    if split == 'all':
        train_videos = cfg.get('videos').get('train', [])
        test_videos = cfg.get('videos').get('test', [])
        videos = [v for pair in zip_longest(train_videos, test_videos) for v in pair if v is not None]
    else:
        videos = cfg.get('videos').get(split, [])
    if videos == []:
        raise ValueError(f'No videos found for split: {split}')
    return videos

def run_slurm_array_job(cfg, config_file_path, task_fn):
    all_videos = get_tracking_split_videos(cfg)
    
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    videos_for_task = [vid for i, vid in enumerate(all_videos) if i % task_count == task_id]

    print(f"SLURM task {task_id} handling {len(videos_for_task)} videos: {videos_for_task}")
    run_multiprocessing(cfg, videos_for_task, config_file_path, task_fn=task_fn, cpus_per_task=cpus_per_task)

def run_multiprocessing(cfg, all_videos, config_file_path, task_fn, cpus_per_task=None):
    use_gpus = cfg.get('tracking').get('gpu_accelerate', False)

    available_memory_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)
    mem_per_proc = 64  # in GB
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(',')) if use_gpus else 0

    if use_gpus:
        max_processes = min(num_gpus, int(available_memory_gb // mem_per_proc))
    else:
        available_cpus = multiprocessing.cpu_count() if cpus_per_task is None else cpus_per_task
        max_processes = min(available_cpus, int(available_memory_gb // mem_per_proc))

    max_processes = min(max_processes, len(all_videos))
    if max_processes < 1:
        print("Insufficient resources to run even one process. Exiting.")
        sys.exit(1)

    print(f"Configuring pool with max {max_processes} concurrent processes with {'GPU' if use_gpus else 'CPU'}.")

    if max_processes > 1:
        with Manager() as manager:
            gpu_queue = manager.Queue()
            if use_gpus:
                print(f"Initialising shared GPU queue with {num_gpus} GPUs.")
                for gpu_id in range(num_gpus):
                    gpu_queue.put(gpu_id)

            task_arguments = []
            for video_id in all_videos:
                task_arguments.append((task_fn, gpu_queue, use_gpus, video_id))

            print(f"Starting processing {len(all_videos)} videos with a pool of {max_processes} workers.")
            
            with multiprocessing.Pool(
                processes=max_processes,
                initializer=init_worker_config,
                initargs=(config_file_path,)
            ) as pool:
                pool.starmap(pool_worker_task, task_arguments)
    else:
        init_worker_config(config_file_path)
        with Manager() as manager:
            gpu_queue = manager.Queue()
            if use_gpus:
                gpu_queue.put(0)
            print(f"Starting processing {len(all_videos)} videos with a pool of {max_processes} workers.")
            for video_id in all_videos:
                pool_worker_task(task_fn, gpu_queue, use_gpus, video_id)

    print("All videos processed.")


def run_with_resources(cfg, config_file_path, task_fn):
    multiprocessing.set_start_method('spawn', force=True)
    if 'SLURM_ARRAY_TASK_ID' in os.environ and 'SLURM_ARRAY_TASK_COUNT' in os.environ:
        print("Detected SLURM array job")
        run_slurm_array_job(cfg, config_file_path, task_fn)
    else:
        print("Running locally with multiprocessing")
        all_videos = get_tracking_split_videos(cfg)
        run_multiprocessing(cfg, all_videos, config_file_path, task_fn=task_fn)

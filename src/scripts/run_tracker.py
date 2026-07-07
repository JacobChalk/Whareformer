import argparse
import traceback

from utils.launcher import run_with_resources

def process_video(cfg, video_id, gpu_id=None):
    from tracking import VideoInstance
    if gpu_id is not None:
        print(f"Starting processing video {video_id} on GPU {gpu_id}")
    else:
        print(f"Starting processing video {video_id} on CPU")
    video_instance = None
    try:
        tracking_params = cfg.get('tracking')
        model_params = cfg.get('model')
        tracking_params.update({'model': model_params, 'device_id': gpu_id})
        print(f"[{video_id}] Initialising VideoInstance...")
        video_instance = VideoInstance(cfg, video_id, tracking_params=tracking_params)
        print(f"[{video_id}] Starting tracking...")
        video_instance.track()
        print(f"[{video_id}] Tracking completed successfully.")
    except Exception as e:
        print(f"[{video_id}] Exception occurred during processing:")
        print(traceback.format_exc())
        print(f"[{video_id}] Failed processing: {e}")
    finally:
        if video_instance is not None:
            try:
                print(f"[{video_id}] Shutting down VideoInstance...")
                video_instance.shutdown()
                print(f"[{video_id}] Shutdown complete.")
            except Exception as shutdown_e:
                print(f"Error during shutdown: {shutdown_e}")
        print(f"[{video_id}] Finished processing video {video_id}\n{'-'*50}")

def run_tracker(cfg):
    run_with_resources(cfg, cfg.config_file, task_fn=process_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/osnom_config.yaml", help="Path to config file")
    args = parser.parse_args()
    print(f"Initialising Config...")
    from utils import Config
    cfg = Config(args.config)
    print(f"Running Tracker...")
    run_tracker(cfg)


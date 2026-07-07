import traceback
import argparse

from utils.launcher import run_with_resources

def process_video(cfg, video_id, gpu_id=None):
    print(f"Starting processing video {video_id} on GPU {gpu_id}")
    video_instance = None
    try:
        dataset = cfg.get('dataset')
        if dataset == 'epic':
            from feature_extraction.extractors.epic_extractor import EPICFeatureExtractor
            video_instance = EPICFeatureExtractor(video_id, cfg)
        elif dataset == 'hd_epic':
            from feature_extraction.extractors.hd_epic_extractor import HDEPICFeatureExtractor
            video_instance = HDEPICFeatureExtractor(video_id, cfg)
        elif dataset == 'it3dego':
            from feature_extraction.extractors.it3dego_extractor import IT3DEgoFeatureExtractor
            video_instance = IT3DEgoFeatureExtractor(video_id, cfg)
        else:
            raise ValueError(f"Unknown dataset configuration: {dataset}")

        print(f"[{video_id}] Running feature extraction...")
        video_instance.run()
        
    except Exception as e:
        print(f"[{video_id}] Exception occurred during processing:")
        print(traceback.format_exc())
    finally:
        if video_instance is not None:
            video_instance.shutdown()
        print(f"[{video_id}] Finished processing video {video_id}\n{'-'*50}")

def extract_features(cfg):
    run_with_resources(cfg, cfg.config_file, task_fn=process_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/feature_extraction_config.yaml", help="Path to config file")
    args = parser.parse_args()
    print(f"Initialising Config...")
    from utils import Config
    cfg = Config(args.config)
    print(f"Extracting Features...")
    extract_features(cfg)

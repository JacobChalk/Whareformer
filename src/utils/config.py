import pandas as pd
import yaml
import os
from typing import Any

class Config:
    def __init__(self, config_file: str):
        super(Config, self).__init__()
        self.config_file = config_file
        self._config_data = self._load_config()
        self._vid_info_df = None
        self._video_data_cache = {}

    def _load_config(self) -> Any:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")

        with open(self.config_file, "r") as f:
            return yaml.safe_load(f)
        
    def _get_vid_info_df(self) -> pd.DataFrame:
        if self._vid_info_df is None:
            csv_path = self.get("paths.video_info_file")
            print(f"[Config PID {os.getpid()}] Loading CSV: {csv_path}")
            self._vid_info_df = pd.read_csv(csv_path)
        return self._vid_info_df
    
    def get_video_data(self, video_id: str) -> dict:
        if video_id in self._video_data_cache:
            return self._video_data_cache[video_id]

        vid_info_df = self._get_vid_info_df()
        
        try:
            row = vid_info_df[vid_info_df['video_id'] == video_id].iloc[0]
        except IndexError:
            raise ValueError(f"Video ID {video_id} not found in {self.get('paths.video_info_file')}")

        fps = int(row['fps'])
        num_frames = int(row['frames'])
        
        video_data = {
                "num_frames": num_frames,
                "fps": 60 if fps == 59 else fps
            }
        
        self._video_data_cache[video_id] = video_data
        return video_data

    def get(self, key: str, default_value : Any = {}) -> Any:
        if key == "":
            return default_value
        keys = key.split(".")
        value = self._config_data
        for k in keys:
            if not isinstance(value, dict):
                return default_value
            value = value.get(k, None)
        return value if value is not None else default_value
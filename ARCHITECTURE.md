# Architecture

A quick explanation of how the code executes, to complement the [README](README.md).

## One instance per video

Feature extraction and tracking both process each video independently. There is no shared state between videos, so they can run concurrently in separate processes.

- **Feature extraction** spawns a dataset-specific extractor (`EPICFeatureExtractor`, `HDEPICFeatureExtractor`, or `IT3DEgoFeatureExtractor`, chosen based on `cfg['dataset']`), since each raw dataset format is different.
- **Tracking** spawns a single generic `VideoInstance`, since by this point all datasets have already been standardised into the same feature format. This is why the tracking entry script has no dataset branching.

Both entry scripts follow the same pattern: build the instance, run it, and always call `shutdown()` in a `finally` block. Exceptions are caught and logged per-video rather than crashing the whole job, so one bad video doesn't take down a large batch run.

## Parallelism

This is handled by `run_with_resources` in `utils/launcher.py`, which is shared between feature extraction and tracking:

- If SLURM array env vars are present, it shards the video list across array tasks (`i % task_count == task_id`), then runs a local multiprocessing pool within each task.
- Otherwise, it just runs a local multiprocessing pool directly.

The number of concurrent processes is capped by whichever is smallest: available GPUs (or CPUs), available RAM (assuming ~64GB per process), or the number of videos left to process. If GPUs are used, workers pull a free GPU id from a shared queue and return it when done (even on failure), so processing naturally throttles to the number of available GPUs. You may need to adapt this script to suit your specific resources.

## Adding a new dataset

1. Write a new extractor and dataset class.
2. Add a branch for it in `process_video` in `extract_features.py`.
3. Make sure it writes features in the expected output format detailed below.

### Expected output format

Each extractor must produce:

- A `video_info.csv` with 3 columns: video id, number of frames, and fps. This is not handled by the extractor by default but is nonetheless required.
- Per video, the following files under `<ROOT>/<VIDEO-ID>/`:
  - `observations.json` — per-frame object annotations, structured as:
    ```json
    {
      "video_annotations": [
        {
          "frame_name": "frame_0000000283",
          "annotations": [
            {"id": "e44a21e6e04d424c", "name": "mug", "bounding_box": [693, 847, 775, 979]}
          ]
        }
      ]
    }
    ```
  - `2D_feat_masked.pkl` — dict mapping each frame name to an `(N, D)` array of 2D features, one row per annotation in that frame.
  - `3D_feat_aligned.pkl` — dict mapping each frame name to an `(N, 3)` array of 3D locations, one row per annotation in that frame.

`N` must match across all three files for a given frame, and rows must correspond to the same objects in the same order e.g. if a frame has 2 annotations in `observations.json`, both `.pkl` files must have shape `(2, D)` and `(2, 3)` for that frame.

### Swapping the feature backbone

Backbones are defined in `feature_extraction/utils/models.py` and selected by name (e.g. `dinov2_giant`) via `load_2d_model`. To add one, implement a transform + forward pass (see `get_dino_style_transform` / `forward_dino_model_style` for the DINOv2 example) and add a branch for it in `load_2d_model`.
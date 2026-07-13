<div align="center">

# Whareformer: Learning to Track What is Where in Long Egocentric Videos

**Jacob Chalk, Saptarshi Sinha, Dima Damen, Yannis Kalantidis, Diane Larlus**

**ECCV 2026**

[![Project Page](https://img.shields.io/badge/Project-Webpage-blue?style=for-the-badge&logo=github)](https://jacobchalk.github.io/Whareformer/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2607.08537)
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-4b44ce?style=for-the-badge)](https://eccv.ecva.net)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-cu126-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/jacobchalk/Whareformer?style=social)](https://github.com/jacobchalk/Whareformer/stargazers)

</div>

---

**Whareformer** is an online, long-term 3D object tracker for egocentric video that reasons jointly about *where* an object is and *what* it looks like, enabling long-term tracking which is robust through occlusions, out-of-frame periods, and evolving object appearance across EPIC-KITCHENS, IT3DEgo, and HD-EPIC.

> [Paper](https://arxiv.org/abs/2607.08537) &nbsp;•&nbsp; [Project Webpage](https://jacobchalk.github.io/Whareformer/) &nbsp;•&nbsp; [Architecture](ARCHITECTURE.md) &nbsp;•&nbsp; [Pre-extracted Features](#pre-extracted-features) &nbsp;•&nbsp; [Pre-trained Models](#pre-trained-model)

---

## Table of Contents

- [Citing](#citing)
- [Setup Environment](#setup-environment)
- [Architecture](#architecture)
- [Pre-extracted Features](#pre-extracted-features)
- [Pre-trained Model](#pre-trained-model)
- [Training and Evaluating Whareformer](#training-and-evaluating-whareformer)
- [License](#license)

---

## Citing

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@InProceedings{chalk2026whareformer,
    title     = {Whareformer: Learning to Track What is Where in Long Egocentric Videos},
    author    = {Chalk, Jacob and Sinha, Saptarshi and Damen, Dima and Kalantidis, Yannis and Larlus, Diane},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2026}
}
```

---

## Setup Environment

### Option A - Latest packages

```bash
conda create -n whareformer python=3.13
conda activate whareformer
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install pandas pyyaml pycolmap lmdb trimesh pyrender opencv-python decord scipy scikit-learn tqdm transformers xformers wandb 'git+https://github.com/cheind/py-motmetrics.git'
conda install conda-forge::open3d # If reconstructing EPIC meshes, otherwise not required
export PYTHONPATH=/path/to/Whareformer/src:$PYTHONPATH
```

### Option B - Match our exact package versions (recommended for reproducibility)

```bash
conda create -n whareformer python=3.13.13
conda activate whareformer
conda install conda-forge::open3d=0.19.0 # If reconstructing EPIC meshes, otherwise not required
pip3 install -r requirements.txt
```

---

## Architecture

We provide details on how the codebase generally operates in [Architecture](ARCHITECTURE.md).

## Pre-extracted Features

We provide pre-extracted features for all observations in **EPIC**, **IT3DEgo**, and **HD-EPIC**, along with the training data LMDB, **[here](https://uob-my.sharepoint.com/:f:/g/personal/jc17360_bristol_ac_uk/IgA46O5x-kvGT4CqSfdx1DWCAUmq1H2IWLoN1qWGtzj9au0?e=vVBla6)**. Downloading these is recommended for reproducibility and ease of use.

If you wish to use the raw DINOv2 features without PCA applied, you can download them **[here](https://uob-my.sharepoint.com/:f:/g/personal/jc17360_bristol_ac_uk/IgCab5vZmTyJQ4CeiOUwuiJUAa7aclUjNoOqGcnrsY5NU_4?e=FU9y06)**.

## Pre-trained Model

Our model weights are available **[here](https://uob-my.sharepoint.com/:f:/g/personal/jc17360_bristol_ac_uk/IgBNUQ7Ibh8LRY1nl5diqeyqAYJm8-ctCTzuIPBmwzUl83Q?e=9bWQpI)**. Downloading these weights is recommended to reproduce the results reported in the paper.

---

## Training and Evaluating Whareformer

> Steps 0–3 are for training from scratch; you may need to update the relevant paths in the YAML config files. **Skip to Step 4** if you are using our pre-extracted features and training data.

### Step 0 - Download datasets

> To use your own dataset instead, see [Architecture](ARCHITECTURE.md#adding-a-new-dataset) for the expected output format before tracking.

Please refer to the original dataset pages for potentially more convenient links/methods to download the relevant parts of each dataset.

<details>
<summary><b>0a. EPIC-KITCHENS</b></summary>

Download [EPIC-Kitchens RGB frames](https://github.com/epic-kitchens/epic-kitchens-100-annotations), [VISOR Dense Interpolations](https://data.bris.ac.uk/data/dataset/2v6cgv1x04ol22qp9rm9x2j6a7), [EPIC-Fields Sparse Reconstructions](https://www.dropbox.com/scl/fo/0wtphqqyp4fu6bd7dhbfs/h?rlkey=ju21graeixi6vpecrf7rqurpt), and [VISOR dense frame mapping](https://uob-my.sharepoint.com/:u:/g/personal/jc17360_bristol_ac_uk/IQCvI5tqyE96RZ6EcCmMaYXjAbj_IW1x2TxVhWP0A2z9d3I?e=MZeAfz).

The feature extractor assumes all data is extracted into one folder with the following file structure:

```text
EPIC-Data
├── EPIC-Fields
│   ├── poses
│   │   ├── P01_01.json
│   │   ├── ...
│   │   └── P37_103.json
│   └── sparse
│       ├── P01_01
│       ├── ...
│       └── P37_103
├── EPIC-KITCHENS
│   ├── P01
│   │   └── rgb_frames
│   │       ├── P01_01
│   │       ├── ...
│   │       └── P01_104
│   ├── ...
│   └── P37
├── VISOR
│   ├── P01_01_interpolations.json
│   ├── ...
│   └── P37_103_interpolations.json
└── dense_visor_frame_mapping.json
```
</details>

<details>
<summary><b>0b. IT3DEgo</b></summary>

Download the [dataset](https://drive.google.com/file/d/1VVszWG4mmm0g3ai3EoZw-3cGNBmZCN-9) from the [IT3DEgo](https://github.com/IT3DEgo/IT3DEgo) repository.

The feature extractor assumes all data is extracted into one folder with the following file structure:

```text
IT3DEgo
├── annotations
├── calibrations
├── enrollment_info
└── raw_videos
```
</details>

<details>
<summary><b>0c. HD-EPIC</b></summary>

Download [videos](https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47), [masks](https://www.dropbox.com/scl/fo/f7hwei2m8y3ihlhp669h4/ALM8_1LDETY40O-06-ptr3A?rlkey=yrmqm3zk284htr5yjxb4z5nwp&e=1&st=815ovw6m), and [annotation information (mask_info and assoc_info)](https://github.com/hd-epic/hd-epic-annotations/blob/main/scene-and-object-movements/mask_info.json) from [HD-EPIC](https://hd-epic.github.io/).

The feature extractor assumes all data is extracted into one folder with the following file structure:

```text
HD-EPIC
├── videos
├── hd_epic_association_masks
├── frame_info.json
└── assoc_info.json
```
</details>

### Step 1 - Extract features for each dataset

```bash
python scripts/extract_features.py --config config/epic/feature_extraction_config.yaml
python scripts/extract_features.py --config config/it3dego/feature_extraction_config.yaml
python scripts/extract_features.py --config config/hd_epic/feature_extraction_config.yaml
```

> **Note:** EPIC feature extraction will reconstruct 3D meshes if not already present in the target `scene_dir` directory. This step takes a while, so we have pre-extracted them [here](https://uob-my.sharepoint.com/:u:/g/personal/jc17360_bristol_ac_uk/IQDHGYZj4OaZT7i_ClT546UwAUD3Qmf-HgiyPpOW47lHIz0?e=vOlcD6).

### Step 2 - Learn PCA on training data and apply to all datasets

```bash
python scripts/apply_pca.py \
    --output_dir /path/to/epic/features \
    --config_path config/epic/feature_extraction_config.yaml

python scripts/apply_pca.py \
    --output_dir /path/to/it3dego/features \
    --config_path config/it3dego/feature_extraction_config.yaml \
    --pca_path /path/to/pca_learned_on_epic

python scripts/apply_pca.py \
    --output_dir /path/to/hd_epic/features \
    --config_path config/hd_epic/feature_extraction_config.yaml \
    --pca_path /path/to/pca_learned_on_epic
```

### Step 3 - Create training data

**3a. Get oracle tracking data**

```bash
python scripts/run_tracker.py --config config/epic/oracle_config.yaml
```

**3b. Build LMDB database for training**

```bash
python scripts/merge_lmdb.py --source_dir /path/to/oracle_training_data --split train
```

### Step 4 - Train model

```bash
python training/train_model.py --config config/epic/whareformer_config.yaml
```

### Step 5 - Run tracking inference

```bash
python scripts/run_tracker.py --config config/epic/whareformer_config.yaml
```

### Step 6 - Evaluate results

```bash
python scripts/evaluate.py --results_dir /path/to/tracking_outputs
```

---

## License

This project is released under the **[MIT License](LICENSE)**.

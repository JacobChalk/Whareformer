#! /bin/bash

# Usage: reconstruct_mesh.sh <video_id> <cameras_path> <image_path> <dense3d_path>
set -eu

VIDEO_ID=$1
CAMERAS_PATH=$2
IMAGE_PATH=$3
DENSE3D_PATH=$4

# Resolve the directory where this bash script is located
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_UTILS_SCRIPT="$SCRIPT_DIR/colmap_o3d_utils.py"

# =========
# Hyperparameters
MIN_PIX=10
# =========

FINAL_MESH_PATH="$DENSE3D_PATH/mvs_mesh.ply"

# 0. Early Exit
if [ -f "$FINAL_MESH_PATH" ]; then
    echo "Final mesh already exists for $VIDEO_ID at $FINAL_MESH_PATH. Exiting."
    exit 0
fi

mkdir -p "$DENSE3D_PATH"

# 1. Undistort images
echo "1. Undistorting images..."
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$CAMERAS_PATH" \
    --output_path "$DENSE3D_PATH" \
    --output_type COLMAP

# 1.5. Extract depth bounds
echo "Computing depth bounds from sparse reconstruction..."
read DEPTH_MIN DEPTH_MAX < <(python3 "$PYTHON_UTILS_SCRIPT" depth_bounds --reconstruction "$CAMERAS_PATH" --margin 0.25)
echo "   Depth bounds: min=$DEPTH_MIN max=$DEPTH_MAX (COLMAP units)"

# 2. PatchMatch Stereo
echo "2. Running PatchMatch Stereo..."
colmap patch_match_stereo \
    --workspace_path "$DENSE3D_PATH" \
    --workspace_format COLMAP \
    --PatchMatchStereo.depth_min $DEPTH_MIN \
    --PatchMatchStereo.depth_max $DEPTH_MAX \
    --PatchMatchStereo.cache_size 64

# 3. Stereo Fusion
echo "3. Fusing Multi-View Depth Maps (min_pix=$MIN_PIX)..."
colmap stereo_fusion \
    --workspace_path "$DENSE3D_PATH" \
    --workspace_format COLMAP \
    --input_type geometric \
    --StereoFusion.min_num_pixels $MIN_PIX \
    --StereoFusion.cache_size 64 \
    --output_path "$DENSE3D_PATH/fused.ply"

# 4. Delaunay Meshing (The Geometric Bounding Filter)
echo "4. Running Delaunay Meshing for Outlier Removal..."
colmap delaunay_mesher \
    --input_path "$DENSE3D_PATH" \
    --DelaunayMeshing.max_proj_dist 5 \
    --DelaunayMeshing.quality_regularization 5 \
    --output_path "$DENSE3D_PATH/mvs_delaunay_mesh.ply"

# 5. Post-Processing: Island Removal & Taubin Smoothing
echo "5. Refining final mesh (removing islands and smoothing)..."
python3 "$PYTHON_UTILS_SCRIPT" refine --mesh "$DENSE3D_PATH/mvs_delaunay_mesh.ply" --output "$FINAL_MESH_PATH"

echo "Reconstruction complete for $VIDEO_ID."
import numpy as np
import cv2
import torch
import pycolmap

from pathlib import Path
from tqdm import tqdm

# --- 3D Geometry Utilities ---

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])

def colmap_pose_to_c2w(qvec, tvec, scale=1.0):
    R = qvec2rotmat(qvec)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ (tvec * scale)
    return c2w

def get_bounding_box_center(box):
    min_x, min_y, max_x, max_y = box
    center_x = min_x + (max_x - min_x) / 2.0
    center_y = min_y + (max_y - min_y) / 2.0
    return center_x, center_y

def rescale_point(point, mask_res, frame_res_w, frame_res_h):
    x, y = point
    mask_res_w, mask_res_h = mask_res
    
    scale_x = frame_res_w / mask_res_w
    scale_y = frame_res_h / mask_res_h
    
    tx = int(round(x * scale_x))
    ty = int(round(y * scale_y))
    
    tx = max(0, min(tx, frame_res_w - 1))
    ty = max(0, min(ty, frame_res_h - 1))
    
    return tx, ty

def lift_to_3d(camera_pose,
               camera_intrinsics,
               point,
               depth,
               object_mask,
               mask_res):

    frame_res_w, frame_res_h, fx, fy, cx, cy = camera_intrinsics

    tx, ty = rescale_point(
        point,
        mask_res,
        frame_res_w,
        frame_res_h
    )

    object_mask_resized = cv2.resize(
        object_mask.astype(np.uint8),
        (frame_res_w, frame_res_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Keep only object pixels
    valid_depths = depth[object_mask_resized]

    if valid_depths.size > 0:
        Z = float(np.median(valid_depths))
    else: # Centroid fallback
        Z = float(depth[ty, tx])
    Z = max(Z, 1e-6)

    x = Z * (tx - cx) / fx
    y = Z * (ty - cy) / fy

    uv_norm = np.array([[x, y, Z, 1.0]])

    pred_t = (camera_pose @ uv_norm.T).T

    return pred_t / pred_t[:, 3:]

def lift_object_centroid(object_masks,
                         bounding_boxes,
                         camera_pose,
                         aligned_depth,
                         camera_intrinsics,
                         mask_res):
    loc_3d = []

    for bbox, object_mask in zip(bounding_boxes, object_masks):

        center = get_bounding_box_center(bbox)

        location_estiamte = lift_to_3d(
            camera_pose=camera_pose,
            camera_intrinsics=camera_intrinsics,
            point=center,
            depth=aligned_depth,
            object_mask=object_mask,
            mask_res=mask_res
        )

        loc_3d.append(location_estiamte)

    loc_3d_np = np.stack(loc_3d, axis=0).reshape(len(loc_3d), 4)

    return loc_3d_np[:, :3]

def compute_metric_scale(
    reconstruction_path: str,
    forward_fn_3d,
    frames_path: str,
    batch_size: int,
    patch_radius: int = 5,
    trim_ratio: float = 0.10,
) -> float:
    print(f"[Scale Recovery] Loading sparse COLMAP reconstruction from {reconstruction_path}")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    print(f"[Scale Recovery] {len(reconstruction.images)} images, "
          f"{len(reconstruction.points3D)} sparse points")

    valid_images = []
    for _, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        frame_stem = Path(image.name).stem
        frame_path = Path(frames_path) / f"{frame_stem}.jpg"
        observations = [
            (p2d.point3D_id, p2d.x(), p2d.y())
            for p2d in image.points2D
            if p2d.has_point3D()
        ]
        if len(observations) == 0:
            continue
        valid_images.append((image, frame_path, observations))

    if not valid_images:
        raise RuntimeError(
            "[Scale Recovery] No valid images found. "
            "Check reconstruction_path, frames_path, and frame_names."
        )

    print(f"[Scale Recovery] Using {len(valid_images)} frames")
    scale_samples = []

    for i in tqdm(range(0, len(valid_images), batch_size)):
        chunk = valid_images[i:i+batch_size]
        frames_t = torch.stack([
            torch.from_numpy(cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB))
                .permute(2,0,1).float() / 255.0
            for _, fp, _ in chunk
        ])
        _, _, H, W = frames_t.shape
        with torch.no_grad():
            depth_batch = forward_fn_3d(frames_t).cpu().numpy()
        
        for j, (image, frame_path, observations) in enumerate(chunk):
            depth_m = depth_batch[j]

            cam_from_world = image.cam_from_world()

            for point3D_id, px, py in observations:
                pt3d = reconstruction.points3D[point3D_id]
                pt_cam = cam_from_world * pt3d.xyz
                z_colmap = float(pt_cam[2])
                if z_colmap <= 1e-3:
                    continue

                xi = int(np.clip(round(px), 0, W - 1))
                yi = int(np.clip(round(py), 0, H - 1))
                patch = depth_m[
                    max(0, yi - patch_radius): min(H, yi + patch_radius + 1),
                    max(0, xi - patch_radius): min(W, xi + patch_radius + 1),
                ]
                valid_px = patch[patch > 1e-3]
                if valid_px.size == 0:
                    continue

                scale_samples.append(float(np.median(valid_px)) / z_colmap)

    if not scale_samples:
        raise RuntimeError("[Scale Recovery] Zero valid scale observations collected.")

    samples = np.array(scale_samples)
    lo = np.percentile(samples, trim_ratio * 100)
    hi = np.percentile(samples, (1 - trim_ratio) * 100)
    inliers = samples[(lo <= samples) & (samples <= hi)]

    scale = round(float(np.median(inliers)), 10)
    print(f"[Scale Recovery] Scale: {scale:.10f} | "
          f"Inliers: {len(inliers):,} / {len(samples):,} | "
          f"Std: {inliers.std():.5f}")
    return scale

# --- Depth Utilities ---

def align_metric_and_scene_depth(metric_depth, scene_depth, mask):
    metric_depth_masked = metric_depth[mask]
    scene_depth_masked = scene_depth[mask]

    if metric_depth_masked.size == 0 or scene_depth_masked.size == 0:
        return metric_depth # Return unaligned

    # Solve for metric_depth_masked * alpha + beta = scene_depth_masked
    A = np.stack([metric_depth_masked, np.ones_like(metric_depth_masked)], axis=-1)
    
    alpha, beta = np.linalg.lstsq(A, scene_depth_masked, rcond=None)[0]
    return metric_depth * alpha + beta
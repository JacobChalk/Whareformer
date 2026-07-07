import open3d as o3d
import numpy as np
import argparse
import sys

def refine_mesh(mesh_path: str, output_path: str, smooth_iterations: int = 20, min_triangles: int = 1000, size_ratio: float = 0.01):
    print(f"[O3D] Loading final mesh for refinement: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    print("[O3D] Running Connected Component analysis to remove noise blobs...")
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        largest_cluster_size = cluster_n_triangles[largest_cluster_idx]
        
        dynamic_threshold = int(largest_cluster_size * size_ratio)
        cutoff = max(min_triangles, dynamic_threshold)
        
        valid_cluster_ids = np.where(
            (cluster_n_triangles >= cutoff) | (np.arange(len(cluster_n_triangles)) == largest_cluster_idx)
        )[0]
        
        triangles_to_remove = ~np.isin(triangle_clusters, valid_cluster_ids)
        
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        
        total_clusters = len(cluster_n_triangles)
        kept_clusters = len(valid_cluster_ids)
        removed_clusters = total_clusters - kept_clusters
        
        print(f"[O3D] Kept {kept_clusters} valid structures. Deleted {removed_clusters} blobs.")
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    print(f"[O3D] Applying Taubin smoothing ({smooth_iterations} iterations)...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path, mesh)
    print("[O3D] Mesh refinement complete.")

def compute_depth_bounds(
    reconstruction_path: str,
    margin: float = 0.25,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    fallback_min: float = 0.01,
    fallback_max: float = 50.0,
):
    import pycolmap

    def _log(msg):
        print(f"[O3D] {msg}", file=sys.stderr)

    _log(f"Loading sparse reconstruction from {reconstruction_path}...")
    try:
        recon = pycolmap.Reconstruction(reconstruction_path)
    except Exception as e:
        _log(f"Failed to load reconstruction: {e}. Using fallback bounds.")
        print(f"{fallback_min:.6f} {fallback_max:.6f}")
        return

    _log(f"{len(recon.images)} images, {len(recon.points3D)} sparse points.")

    depths = []
    for _, image in recon.images.items():
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world()
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pt3d = recon.points3D[p2d.point3D_id]
            z = float((cam_from_world * pt3d.xyz)[2])
            if z > 0:
                depths.append(z)

    if not depths:
        _log("No valid depth observations found. Using fallback bounds.")
        print(f"{fallback_min:.6f} {fallback_max:.6f}")
        return

    d = np.array(depths)
    lo = np.percentile(d, percentile_lo)
    hi = np.percentile(d, percentile_hi)
    pad = (hi - lo) * margin

    depth_min = max(fallback_min, lo - pad)
    depth_max = hi + pad

    _log(
        f"Depth distribution: p{percentile_lo:.0f}={lo:.4f}, p{percentile_hi:.0f}={hi:.4f} "
        f"(COLMAP units) → bounds [{depth_min:.4f}, {depth_max:.4f}] with {margin*100:.0f}% margin."
    )

    print(f"{depth_min:.6f} {depth_max:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open3D Utilities for COLMAP Mesh Post-Processing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    refine_parser = subparsers.add_parser("refine", help="Remove islands and smooth final mesh")
    refine_parser.add_argument("--mesh", type=str, required=True)
    refine_parser.add_argument("--output", type=str, required=True)

    bounds_parser = subparsers.add_parser("depth_bounds", help="Compute PatchMatch depth bounds from sparse reconstruction")
    bounds_parser.add_argument("--reconstruction", type=str, required=True, help="Path to COLMAP sparse reconstruction directory")
    bounds_parser.add_argument("--margin",         type=float, default=0.25,  help="Fractional padding beyond the percentile range")
    bounds_parser.add_argument("--percentile-lo",  type=float, default=1.0,   help="Lower percentile for depth range")
    bounds_parser.add_argument("--percentile-hi",  type=float, default=99.0,  help="Upper percentile for depth range")

    args = parser.parse_args()

    if args.command == "refine":
        refine_mesh(args.mesh, args.output)
    elif args.command == "depth_bounds":
        compute_depth_bounds(
            args.reconstruction,
            margin=args.margin,
            percentile_lo=args.percentile_lo,
            percentile_hi=args.percentile_hi,
        )
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import yaml


def level_plane(points):
    """
    Fit a plane to points and compute rotation R and translation t to level it
    """
    points = np.asarray(points)
    centroid = points.mean(axis=0)
    Q = points - centroid
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    target = np.array([0.0, 0.0, 1.0])
    if np.allclose(normal, target):
        R = np.eye(3)
    else:
        axis = np.cross(normal, target)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
    centroid_rot = R @ centroid
    t = np.array([0.0, 0.0, -centroid_rot[2]])
    return R, t


if __name__ == "__main__":
    # -----------------------------
    # Inputs: set yaml_config_path to load ground_points/z_range from a YAML; else edit inline below.
    # pcd_path: path to input point cloud; use .pcd or .ply extension (format is inferred from it).
    # -----------------------------
    yaml_config_path = None  # e.g. r"path/to/plab_4-2_rotated.yaml" to reuse stored config

    pcd_path = r"C:\Users\ianrp\Desktop\Assignments\Fifth\ENPH 479\path_planning\utils\alum_cropped.ply"
    out_path = r"C:\Users\ianrp\Desktop\Assignments\Fifth\ENPH 479\path_planning\utils"

    z_range = [0.05, 1]
    res = 0.05

    ground_points = [
        [19.879532, 14.868885, -0.411075],
        [12.943614, 15.292838, -0.453351],
        [3.509066, 14.561074, -0.502289],
        [1.054852, 10.306832, -0.500349],
        [-0.884856, 2.336245, -0.487467],
        [20.829382, -1.913318, -0.361749],
        [14.977832, -0.860415, -0.394499],
        [3.763721, -2.478472, -0.447223],
        [-4.866464, 3.218725, -0.560281],
        [-16.989567, 0.387906, -0.577696],
        [-11.087035, 15.460577, -0.613111],
        [-20.133097, 23.519316, -0.737958]
    ]

    if yaml_config_path and os.path.isfile(yaml_config_path):
        with open(yaml_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if cfg:
            if "z_range" in cfg:
                z_range = list(cfg["z_range"])
            if "ground_points" in cfg:
                ground_points = [list(p) for p in cfg["ground_points"]]
            if "pcd_path" in cfg:
                pcd_path = cfg["pcd_path"]
            if "out_path" in cfg:
                out_path = cfg["out_path"]
            if "resolution" in cfg:
                res = float(cfg["resolution"])
        print(f"Loaded config from {yaml_config_path}: z_range={z_range}, {len(ground_points)} ground points")

    os.makedirs(out_path, exist_ok=True)

    # -----------------------------
    # Load point cloud (PCD or PLY; format from file extension)
    # -----------------------------
    cloud_ext = os.path.splitext(pcd_path)[1].lower()
    if cloud_ext not in (".pcd", ".ply"):
        raise ValueError(
            f"Point cloud path must have extension .pcd or .ply, got: {pcd_path}"
        )
    pcd_o3d = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd_o3d.points)
    if pts.size == 0:
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    # -----------------------------
    # Compute leveling + apply
    # NOTE: this matches your existing `pcd_map_conversion.py` logic:
    #   pcd_leveled = (R @ (pcd + t).T).T
    # -----------------------------
    R, t = level_plane(ground_points)
    pts_leveled = (R @ (pts + t).T).T

    # -----------------------------
    # Save rotated point cloud: <name>_rotated.pcd (always PCD output)
    # -----------------------------
    stem = os.path.splitext(os.path.basename(pcd_path))[0]
    rotated_name = f"{stem}_rotated"
    rotated_cloud_path = os.path.join(out_path, f"{rotated_name}.pcd")

    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(pts_leveled)
    if pcd_o3d.has_colors():
        pcd_rot.colors = pcd_o3d.colors
    if pcd_o3d.has_normals():
        normals = np.asarray(pcd_o3d.normals)
        pcd_rot.normals = o3d.utility.Vector3dVector((R @ normals.T).T)
    else:
        # PLY sources often lack normals. Open3D then writes 12 bytes/point (xyz only),
        # but the C++ OccGridBridge loader assumes 24 bytes (xyz + normals). Without
        # dummy normals, the C++ misreads and skips every other point, causing wrong
        # bounds and ~10+ pixel offset between static map and live scan.
        n = len(pts_leveled)
        pcd_rot.normals = o3d.utility.Vector3dVector(np.zeros((n, 3), dtype=np.float64))

    o3d.io.write_point_cloud(rotated_cloud_path, pcd_rot)
    print(f"Saved rotated point cloud: {rotated_cloud_path}")

    # -----------------------------
    # Create occupancy map PNG + YAML from rotated points
    # Bounds from ALL leveled points (no z filter). Formula must match C++ OccGridBridge:
    #   w = (int)((x_max - x_min) / res) + 1,  h = (int)((y_max - y_min) / res) + 1
    # Use the same resolution (e.g. 0.05) as in path_planning.launch.py to avoid dimension mismatch.
    # -----------------------------
    x_min, x_max = float(np.min(pts_leveled[:, 0])), float(np.max(pts_leveled[:, 0]))
    y_min, y_max = float(np.min(pts_leveled[:, 1])), float(np.max(pts_leveled[:, 1]))
    w = int((x_max - x_min) / res) + 1
    h = int((y_max - y_min) / res) + 1
    print(f"Grid bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], res={res} -> size {w} x {h} (width x height)")

    grid = np.full((h, w), 255, dtype=np.uint8)  # white=free, black=occupied

    x_idx = ((pts_leveled[:, 0] - x_min) / res).astype(np.int32)
    y_idx = h - 1 - ((pts_leveled[:, 1] - y_min) / res).astype(np.int32)
    mask = (pts_leveled[:, 2] >= z_range[0]) & (pts_leveled[:, 2] <= z_range[1])
    x_idx, y_idx = x_idx[mask], y_idx[mask]

    valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
    grid[y_idx[valid], x_idx[valid]] = 0  # black = occupied

    png_path = os.path.join(out_path, f"{rotated_name}.png")
    yaml_path = os.path.join(out_path, f"{rotated_name}.yaml")

    plt.imsave(png_path, grid, cmap="gray", origin="upper")

    map_dict = {
        "image": f"{rotated_name}.png",
        "resolution": res,
        "origin": [float(x_min), float(y_min), 0.0],
        "occupied_thresh": 0.6,
        "free_thresh": 0,
        "negate": 0,
        "z_range": z_range,
        "ground_points": ground_points,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(map_dict, f, default_flow_style=None)

    print(f"Saved occupancy map PNG: {png_path}")
    print(f"Saved occupancy map YAML: {yaml_path}")


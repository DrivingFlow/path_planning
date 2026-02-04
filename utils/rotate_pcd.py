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
    # Inputs (edit these as needed)
    # -----------------------------
    pcd_path = r"C:\Users\ianrp\Desktop\Assignments\Fifth\ENPH 479\path_planning\utils\plab_4-1.pcd"
    out_path = r"C:\Users\ianrp\Desktop\Assignments\Fifth\ENPH 479\path_planning\utils"

    # Match defaults in `src/path_planner_node.cpp`
    z_range = [0.1, 2.0]
    res = 0.05

    # -----------------------------
    # Ground points (provided)
    # -----------------------------
    ground_points = [
        [-1.023752, -4.870776, -0.370789],
        [-7.000184, -3.710128, -0.528541],
        [-5.459779, 14.280241, -0.942438],
        [2.564512, 7.997607, -0.598214],
        [-4.150812, 0.827941, -0.548578],
        [-3.162940, 5.527684, -0.570391],
    ]

    os.makedirs(out_path, exist_ok=True)

    # -----------------------------
    # Load full PCD
    # -----------------------------
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
    # Save rotated PCD: <name>_rotated.pcd
    # -----------------------------
    stem = os.path.splitext(os.path.basename(pcd_path))[0]
    rotated_name = f"{stem}_rotated"
    rotated_pcd_path = os.path.join(out_path, f"{rotated_name}.pcd")

    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(pts_leveled)
    if pcd_o3d.has_colors():
        pcd_rot.colors = pcd_o3d.colors
    if pcd_o3d.has_normals():
        normals = np.asarray(pcd_o3d.normals)
        pcd_rot.normals = o3d.utility.Vector3dVector((R @ normals.T).T)

    o3d.io.write_point_cloud(rotated_pcd_path, pcd_rot)
    print(f"Saved rotated point cloud: {rotated_pcd_path}")

    # -----------------------------
    # Create occupancy map PNG + YAML from rotated points
    # -----------------------------
    x_min, x_max = np.min(pts_leveled[:, 0]), np.max(pts_leveled[:, 0])
    y_min, y_max = np.min(pts_leveled[:, 1]), np.max(pts_leveled[:, 1])
    w = int((x_max - x_min) / res) + 1
    h = int((y_max - y_min) / res) + 1

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
        "free_thresh": 0.3,
        "negate": 0,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(map_dict, f, default_flow_style=None)

    print(f"Saved occupancy map PNG: {png_path}")
    print(f"Saved occupancy map YAML: {yaml_path}")


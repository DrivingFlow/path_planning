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
    pcd_path = r"C:\Users\ianrp\Desktop\Assignments\Fifth\ENPH 479\occupancy_gridz\2025-01-29_data1\plab_4.pcd"
    out_path = r"C:\Users\ianrp\Desktop\Assignments\Fifth\ENPH 479\occupancy_gridz\2025-01-29_data1"
    z_range = [0.1, 2.0]
    res = 0.05
    map_name = "2025-01-29_plab_4"

    # Load full PCD
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.asarray(pcd.points)

    # Ground points for leveling
    ground_points = [
        [-1.023752, -4.870776, -0.370789],
        [-7.000184,-3.710128,-0.528541],
        [-5.459779,14.280241,-0.942438],
        [2.564512,7.997607,-0.598214],
        [-4.150812,0.827941,-0.548578],
        [-3.162940, 5.527684,-0.570391]
    ]

    # Compute leveling
    R, t = level_plane(ground_points)
    pcd_leveled = (R @ (pcd + t).T).T

    # -------------------------------------------------------
    # Compute bounding box from all points
    # -------------------------------------------------------
    x_min, x_max = np.min(pcd_leveled[:, 0]), np.max(pcd_leveled[:, 0])
    y_min, y_max = np.min(pcd_leveled[:, 1]), np.max(pcd_leveled[:, 1])
    w = int((x_max - x_min) / res) + 1
    h = int((y_max - y_min) / res) + 1

    # -------------------------------------------------------
    # Create empty grid (all free initially)
    # -------------------------------------------------------
    grid = np.full((h, w), 255, dtype=np.uint8)  # white=free, black=occupied

    # Project points into grid and mark occupancy if within z_range
    x_idx = ((pcd_leveled[:, 0] - x_min)/res).astype(np.int32)
    y_idx = h - 1 - ((pcd_leveled[:, 1] - y_min)/res).astype(np.int32)
    mask = (pcd_leveled[:, 2] >= z_range[0]) & (pcd_leveled[:, 2] <= z_range[1])
    x_idx, y_idx = x_idx[mask], y_idx[mask]

    # Ensure indices are within bounds
    valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
    grid[y_idx[valid], x_idx[valid]] = 0  # black = occupied

    save_png = False
    save_3ch = True  # set to False to disable saving the 3-channel file
    # Save PNG and YAML
    if save_png:
        plt.imsave(os.path.join(out_path, f"{map_name}.png"), grid, cmap="gray", origin="upper")

        map_dict = {
            "image": f"{map_name}.png",
            "resolution": res,
            "origin": [float(x_min), float(y_min), 0.0],
            "occupied_thresh": 0.6,
            "free_thresh": 0.3,
            "negate": 0
        }

        with open(os.path.join(out_path, f"{map_name}.yaml"), "w") as f:
            yaml.dump(map_dict, f, default_flow_style=None)

        print(f"Saved occupancy map: {map_name}.png")

    # Optional: save a 3-channel representation (X, Y, occupancy) as a new file
    # First channel: real X coordinate for each pixel
    # Second channel: real Y coordinate for each pixel
    # Third channel: occupancy value (0=occupied, 255=free) as in `grid`
    if save_3ch:
        # X increases with column index j; Y increases with row index from bottom to top
        # Note: grid indexing uses origin at upper-left, so convert accordingly
        # For pixel at (i, j) in grid (i row, j col):
        # x = x_min + j * res
        # y = y_min + (h - 1 - i) * res
        j_coords = np.arange(w)
        i_coords = np.arange(h)
        x_vals = x_min + j_coords * res
        y_vals = y_min + (h - 1 - i_coords) * res

        X = np.tile(x_vals[np.newaxis, :], (h, 1))
        Y = np.tile(y_vals[:, np.newaxis], (1, w))

        # Stack into a float32 array: (h, w, 3)
        three_ch = np.zeros((h, w, 3), dtype=np.float32)
        three_ch[:, :, 0] = X
        three_ch[:, :, 1] = Y
        three_ch[:, :, 2] = grid.astype(np.float32)

        out_3ch_path = os.path.join(out_path, f"{map_name}_3ch.npy")
        np.save(out_3ch_path, three_ch)
        print(f"Saved 3-channel occupancy array: {os.path.basename(out_3ch_path)}")
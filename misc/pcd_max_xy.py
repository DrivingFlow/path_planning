import os
import open3d as o3d
import numpy as np


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pcd_path = os.path.join(repo_root, "utils", "plab_4-1_rotated.pcd")

    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise RuntimeError(f"Empty point cloud or failed to load: {pcd_path}")

    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))

    print(f"PCD: {pcd_path}")
    print(f"Min x: {x_min}")
    print(f"Max x: {x_max}")
    print(f"Min y: {y_min}")
    print(f"Max y: {y_max}")


if __name__ == "__main__":
    main()

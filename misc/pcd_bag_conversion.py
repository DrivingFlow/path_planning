
'''
    This script does several things
    1. reads a bag and converts it into a pkl file for visualization
        TODO: apply levelling and convert to occupancy grid for visuallization
        TODO: Add in map occupancy grid conversion and overlay eventually
'''

import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from PIL import Image
import cv2

class OccupancyGridProcessor:
    def __init__(self, bag_path, topic, vanilla_pcd_path, ground_points):
        """
        Parameters:
        -----------
        bag_path : str or Path
            Path to the ROS2 bag file containing live scans.
        topic : str
            ROS2 topic name for the pointcloud2 messages.
        vanilla_pcd_path : str or Path
            Path to the vanilla environment PCD file.
        ground_points : list of [x,y,z]
            Manually selected points to define the ground plane for leveling.
        """
        self.bag_path = Path(bag_path)
        self.topic = topic
        self.vanilla_pcd_path = Path(vanilla_pcd_path)
        self.ground_points = np.array(ground_points, dtype=np.float32)

        # Load vanilla PCD for bounding box and alignment
        self.vanilla_points = self.load_pcd(self.vanilla_pcd_path)

        # Compute leveling rotation and translation using the ground points
        self.R, self.t = self.level_plane(self.ground_points)

    def load_pcd(self, pcd_path):
        """
        Load a PCD file and return Nx3 NumPy array of points.
        
        Parameters:
        -----------
        pcd_path : str or Path
            Path to the PCD file.
            
        Returns:
        --------
        points : np.ndarray of shape (N,3)
        """
        pcd = o3d.io.read_point_cloud(str(pcd_path))  # read the PCD
        points = np.asarray(pcd.points, dtype=np.float32)
        return points

    @staticmethod
    def level_plane(points):
        """
        Compute rotation (R) and translation (t) to level a plane to z=0
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
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
        centroid_rot = R @ centroid
        t = np.array([0.0, 0.0, -centroid_rot[2]])
        return R, t

    def pointcloud2_to_xyz(self, msg) -> np.ndarray:
        """
        Convert PointCloud2 message to Nx3 NumPy array (x, y, z)
        Vectorized for speed.
        """
        # Number of points
        num_points = msg.width * msg.height
        if num_points == 0:
            return np.zeros((0,3), dtype=np.float32)

        # Create structured dtype from fields
        dtype_list = [(f.name, np.float32) for f in msg.fields if f.name in ('x','y','z')]
        structured_array = np.frombuffer(msg.data, dtype=dtype_list, count=num_points)
        
        # Convert to Nx3 float32 array
        xyz = np.zeros((num_points,3), dtype=np.float32)
        xyz[:,0] = structured_array['x']
        xyz[:,1] = structured_array['y']
        xyz[:,2] = structured_array['z']
        return xyz

    def read_live_scans(self):
        """
        Reads the live scans from the bag file and returns a list of Nx3 arrays
        Optimized for speed using vectorized pointcloud2_to_xyz.
        """
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        clouds = []

        with AnyReader([Path(self.bag_path)], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic == self.topic]
            if not conns:
                raise RuntimeError(f"{self.topic} not found")
            conn = conns[0]

            for idx, (_, _, raw) in enumerate(reader.messages(connections=[conn])):
                # Deserialize message
                msg = reader.deserialize(raw, conn.msgtype)
                # Vectorized conversion
                points = self.pointcloud2_to_xyz(msg)
                clouds.append(points)

                if idx % 100 == 0:
                    print(f"Read message {idx}")

        return clouds

    def load_vanilla_map(self, pcd_path):
        """
        Load the vanilla PCD map and store as Nx3 array
        """
        pcd = o3d.io.read_point_cloud(pcd_path)
        self.vanilla_points = np.asarray(pcd.points)

    def compute_occupancy_grids(self, live_clouds, res=0.05, z_range=[0.1,2.0]):
        """
        Compute occupancy grids for live scans only, using vanilla PCD
        just to determine bounding box and alignment.

        Live points outside the vanilla bounding box are discarded.

        Returns:
            combined_grids: list of grids (live scans only)
            live_only_grids: same as combined_grids
            grid_shape: (h, w), for overlay alignment with edited vanilla PNG
        """
        # Level vanilla points for bounding box only
        vanilla_leveled = (self.R @ self.vanilla_points.T).T + self.t

        # Compute grid bounds from vanilla PCD only
        x_min, x_max = vanilla_leveled[:,0].min(), vanilla_leveled[:,0].max()
        y_min, y_max = vanilla_leveled[:,1].min(), vanilla_leveled[:,1].max()
        w = int((x_max - x_min)/res) + 1
        h = int((y_max - y_min)/res) + 1

        def points_to_grid_occupancy(points):
            """Vectorized occupancy grid where any point in z_range marks cell occupied,
            points outside bounding box are discarded"""
            grid = np.zeros((h, w), dtype=np.uint8)  # free=0

            # Compute indices
            x_idx = ((points[:,0] - x_min)/res).astype(np.int32)
            y_idx = h - 1 - ((points[:,1] - y_min)/res).astype(np.int32)

            # Filter valid indices inside bounding box
            mask = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
            x_idx, y_idx = x_idx[mask], y_idx[mask]
            z = points[:,2][mask]

            # Occupancy condition: ANY point in z_range
            occ_mask = (z >= z_range[0]) & (z <= z_range[1])

            # Mark cells occupied
            grid_indices = y_idx[occ_mask]*w + x_idx[occ_mask]  # flatten
            unique_indices = np.unique(grid_indices)
            grid.flat[unique_indices] = 255
            return grid

        # Level all live clouds
        live_leveled_list = [(self.R @ c.T).T + self.t for c in live_clouds]

        combined_grids = []
        live_only_grids = []

        # Compute occupancy grids for each live scan
        for idx, cloud_leveled in enumerate(live_leveled_list):
            live_grid = points_to_grid_occupancy(cloud_leveled)
            live_only_grids.append(live_grid)
            combined_grids.append(live_grid)  # overlay with edited PNG later

            if idx % 100 == 0:
                print(f"Converted scan {idx}")

        # Return the shape info for overlay on edited PNG
        grid_shape = (h, w)
        return combined_grids, live_only_grids, grid_shape

    def animate_occupancy_grids_with_png(
        self,
        combined_grids,
        live_grids,
        edited_vanilla_png,
        interval=50,
        xlim=None,
        ylim=None,
        save_path=None,
        show=False,   # <-- NEW FLAG
    ):
        """
        Animate 3 side-by-side plots using OpenCV.
        Rendering is disabled by default for speed & memory safety.
        """

        n_frames = len(combined_grids)

        edited_vanilla_png = edited_vanilla_png.astype(np.uint8)

        def crop_frame(frame):
            if xlim is not None:
                frame = frame[:, xlim[0]:xlim[1]]
            if ylim is not None:
                frame = frame[ylim[0]:ylim[1], :]
            return frame

        edited_vanilla_cropped = crop_frame(edited_vanilla_png)

        # Determine frame size once
        sample = crop_frame(np.maximum(combined_grids[0], edited_vanilla_png))
        h, w = sample.shape
        frame_width = w * 3
        frame_height = h

        # Video writer
        if save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = max(1, int(1000 // interval))
            out = cv2.VideoWriter(
                save_path,
                fourcc,
                fps,
                (frame_width, frame_height),
                isColor=True
            )

        for idx, (cg, lg) in enumerate(zip(combined_grids, live_grids)):

            if idx % 100 == 0:
                print(f"Frame {idx}/{n_frames}")

            cg = cg.astype(np.uint8)
            lg = lg.astype(np.uint8)

            # ---- compute ONLY what you need ----
            overlay = np.maximum(cg, edited_vanilla_png)

            overlay = crop_frame(overlay)
            live = crop_frame(lg)

            combined_frame = np.hstack([overlay, live, edited_vanilla_cropped])
            combined_frame_bgr = cv2.cvtColor(
                combined_frame, cv2.COLOR_GRAY2BGR
            )

            # Frame counter
            cv2.putText(
                combined_frame_bgr,
                f"Frame: {idx + 1}/{n_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if save_path is not None:
                out.write(combined_frame_bgr)

            if show:
                cv2.imshow("Occupancy Grid Animation", combined_frame_bgr)
                if cv2.waitKey(interval) == 27:
                    break

        if show:
            cv2.destroyAllWindows()

        if save_path is not None:
            out.release()
            print(f"Animation saved to {save_path}")


    def load_edited_vanilla_map(self, png_path):
        """
        Load edited vanilla map PNG as a NumPy array for animation/overlay.
        Ensures black=occupied (255) and white=free (0) to match live occupancy grids.
        """
        img = Image.open(png_path).convert("L")  # grayscale
        img_arr = np.array(img, dtype=np.uint8)

        # Invert colors: now occupied=255, free=0
        img_arr = 255 - img_arr

        return img_arr
    
    def save_occupancy_frames_to_npy(
        self,
        combined_grids,
        edited_vanilla_png,
        npy_path
    ):
        """
        Save occupancy frames as a memory-mapped .npy array (T x M x N).
        """

        edited_vanilla_png = edited_vanilla_png.astype(np.uint8)

        T = len(combined_grids)
        H, W = edited_vanilla_png.shape

        # Create memory-mapped array ON DISK
        mmap = np.memmap(
            npy_path,
            dtype=np.uint8,
            mode='w+',
            shape=(T, H, W)
        )

        for i, cg in enumerate(combined_grids):
            overlay = np.maximum(cg.astype(np.uint8), edited_vanilla_png)
            mmap[i] = overlay

            if i % 100 == 0:
                print(f"Written frame {i}/{T}")

        mmap.flush()
        del mmap

        print(f"Saved memory-mapped occupancy frames to {npy_path}")



if __name__ == "__main__":
    ground_points = [
        [-1.023752, -4.870776, -0.370789],
        [-7.000184,-3.710128,-0.528541],
        [-5.459779,14.280241,-0.942438],
        [2.564512,7.997607,-0.598214],
        [-4.150812,0.827941,-0.548578],
        [-3.162940, 5.527684,-0.570391]
    ]
    processor = OccupancyGridProcessor(
        bag_path="2025-01-29_data1\data1_mapframe\data1_mapframe_0.db3",
        topic="/lidar_map",
        vanilla_pcd_path="2025-01-29_data1\\plab_4.pcd",
        ground_points=ground_points
    )
    edited_vanilla_png_path = "2025-01-29_data1\\2025-01-29_plab_4.png"

    animation_path = "C:\\Users\\ianrp\\Desktop\\Assignments\\Fifth\\ENPH 479\\occupancy_gridz\\2025-01-29_data1\\data1_mapframe_z_25-20.mp4"
    npy_path = "C:\\Users\\ianrp\\Desktop\\Assignments\\Fifth\\ENPH 479\\occupancy_gridz\\2025-01-29_data1\\data1_mapframe_z_25-20.npy"

    live_clouds = processor.read_live_scans()
    combined_grids, live_grids, _ = processor.compute_occupancy_grids(live_clouds, z_range=[0.25,2])

    edited_png = processor.load_edited_vanilla_map(edited_vanilla_png_path)
    processor.animate_occupancy_grids_with_png(combined_grids, live_grids, edited_png, xlim = (250,800), ylim = (100,800), interval = 50, save_path=animation_path)

    # processor.save_occupancy_frames_to_npy(combined_grids, edited_png, npy_path)















# #### Visualize all scans sequentially #####

# # Load point clouds
# with open("map_fram_v1_0.pkl", "rb") as f:
#     point_clouds = pickle.load(f)

# print(f"Loaded {len(point_clouds)} scans")
# print("First scan shape:", point_clouds[0].shape)

# # Visualization
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="PointCloud Time History")

# # Initialize an empty PointCloud
# pcd = o3d.geometry.PointCloud()

# # Find the first non-empty scan to set initial geometry and view
# for points in point_clouds:
#     if points.shape[0] > 0:
#         pcd.points = o3d.utility.Vector3dVector(points)
#         break

# vis.add_geometry(pcd)

# # Reset view to fit points
# vis.get_render_option().point_size = 1.0
# vis.reset_view_point(True)

# # Animate all scans
# for scan_idx, points in enumerate(point_clouds):
#     if points.shape[0] == 0:
#         continue
#     pcd.points = o3d.utility.Vector3dVector(points)
#     vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#     print(f"Showing scan {scan_idx + 1}/{len(point_clouds)}")

# vis.destroy_window()


##### HISTOGRAM METHOD #####

# # ---- Parameters ----
# res = 0.05                   # same as your grid resolution
# z_range = [-0.25, 1.6]       # z limits for occupancy
# thresholds = [0, 0.1, 0.2, 0.4, 0.6] # fraction of hits to keep
# show_z_slices = True          # whether to collapse z dimension for plotting

# # ---- Load point clouds ----
# with open("cloud_time_history.pkl", "rb") as f:
#     point_clouds = pickle.load(f)

# print(f"Loaded {len(point_clouds)} scans")
# num_scans = len(point_clouds)

# # ---- Determine grid bounds ----
# all_points = np.vstack([pc for pc in point_clouds if pc.shape[0] > 0])
# x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
# y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
# z_min, z_max = z_range[0], z_range[1]

# w = int((x_max - x_min) / res) + 1
# h = int((y_max - y_min) / res) + 1
# d = int((z_max - z_min) / res) + 1

# print(f"Grid shape: (h={h}, w={w}, d={d})")

# # ---- Initialize 3D histogram ----
# hit_hist = np.zeros((h, w, d), dtype=np.uint16)

# # ---- Populate histogram ----
# for idx, pc in enumerate(point_clouds):
#     if pc.shape[0] == 0:
#         continue
    
#     # Clip to z-range
#     pc = pc[(pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)]
    
#     # Convert points to grid indices
#     x_idx = ((pc[:, 0] - x_min) / res).astype(int)
#     y_idx = h - 1 - ((pc[:, 1] - y_min) / res).astype(int)
#     z_idx = ((pc[:, 2] - z_min) / res).astype(int)
    
#     np.add.at(hit_hist, (y_idx, x_idx, z_idx), 1)
    
#     # Progress
#     if (idx + 1) % 100 == 0:
#         print(f"Processed {idx+1}/{num_scans} scans")

# # ---- Collapse along z for a 2D occupancy projection ----
# # Here we take the max along z-axis
# hit_2d = hit_hist.max(axis=2) / num_scans  # normalize to fraction of scans

# # Flatten the 3D histogram to coordinates for plotting
# ys, xs, zs = np.nonzero(hit_hist)  # only non-zero bins
# counts = hit_hist[ys, xs, zs]

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scale point size by hit count
# ax.scatter(xs, ys, zs, c=counts, cmap='hot', s=counts*0.1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Hit Histogram')
# plt.show()

# # ---- Plot occupancy grids for different thresholds ----
# fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5))
# if len(thresholds) == 1:
#     axes = [axes]

# for ax, thr in zip(axes, thresholds):
#     occupancy = np.zeros_like(hit_2d, dtype=np.uint8)
#     occupancy[hit_2d > thr] = 0       # occupied (static)
#     occupancy[hit_2d <= thr] = 255      # free / dynamic removed
    
#     im = ax.imshow(occupancy, cmap='gray', origin='upper')
#     ax.set_title(f"Threshold = {thr}")
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

# # ---- Optional: save final occupancy grid ----
# final_threshold = 0.6
# final_grid = np.zeros_like(hit_2d, dtype=np.uint8)
# final_grid[hit_2d >= final_threshold] = 0
# final_grid[hit_2d < final_threshold] = 255
# np.save("vanilla_occupancy_grid.npy", final_grid)
# print("Saved final occupancy grid to vanilla_occupancy_grid.npy")

##### ROLLING WINDOW METHOD W/ TEMPORAL DOWNSAMPLING

# # ---- Parameters ----
# res = 0.05                   # grid resolution
# z_range = [-0.25, 2.0]       # z limits for occupancy
# thresholds = [2,3,4]      # min consecutive frames to consider cell occupied
# temporal_skip = 3 # only process every nth frame

# # ---- Load point clouds ----
# with open("cloud_time_history.pkl", "rb") as f:
#     point_clouds = pickle.load(f)

# print(f"Loaded {len(point_clouds)} scans")
# num_scans = len(point_clouds)

# # ---- Determine grid bounds ----
# all_points = np.vstack([pc for pc in point_clouds if pc.shape[0] > 0])
# x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
# y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

# nx = int((x_max - x_min) / res) + 1
# ny = int((y_max - y_min) / res) + 1

# print(f"Grid size: nx={nx}, ny={ny}")

# # ---- Initialize grid trackers ----
# longest_streak = np.zeros((ny, nx), dtype=np.uint16)
# current_streak = np.zeros((ny, nx), dtype=np.uint16)

# # ---- Iterate over scans with temporal downsampling ----
# for idx, pc in enumerate(point_clouds):
#     if idx % temporal_skip != 0:
#         continue  # skip this frame

#     # Create binary occupancy for this frame
#     frame_occ = np.zeros((ny, nx), dtype=bool)
    
#     if pc.shape[0] > 0:
#         # filter by z-range
#         pc = pc[(pc[:, 2] >= z_range[0]) & (pc[:, 2] <= z_range[1])]
        
#         x_idx = ((pc[:, 0] - x_min) / res).astype(int)
#         y_idx = ny - 1 - ((pc[:, 1] - y_min) / res).astype(int)
        
#         frame_occ[y_idx, x_idx] = True

#     # Update streaks
#     current_streak[frame_occ] += 1
#     current_streak[~frame_occ] = 0

#     # Update longest streak
#     longest_streak = np.maximum(longest_streak, current_streak)

#     if (idx + 1) % 100 == 0:
#         print(f"Processed {idx+1}/{num_scans} scans (every {temporal_skip}th frame)")

# # ---- Plot 2D occupancy grids at different thresholds ----
# fig, axes = plt.subplots(1, len(thresholds), figsize=(10 * len(thresholds), 10))
# if len(thresholds) == 1:
#     axes = [axes]

# for ax, thr in zip(axes, thresholds):
#     grid = np.zeros_like(longest_streak, dtype=np.uint8)
#     grid[longest_streak >= thr] = 0   # occupied
#     grid[longest_streak < thr] = 255  # free
#     ax.imshow(grid, cmap='gray', origin='upper')
#     ax.set_title(f"Threshold = {thr} consecutive frames")
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

# # ---- Save final "vanilla" occupancy grid ----
# final_threshold = 5
# final_grid = np.zeros_like(longest_streak, dtype=np.uint8)
# final_grid[longest_streak >= final_threshold] = 0
# final_grid[longest_streak < final_threshold] = 255

# np.save("vanilla_occupancy_grid_streak.npy", final_grid)
# print("Saved vanilla occupancy grid using streak method with temporal downsampling")


##### TEMPORAL DIFFERENCING WITH TEMPORAL DOWNSAMPLING

# # ---- Parameters ----
# res = 0.05                   # grid resolution
# z_range = [-0.25, 2.0]       # z limits for occupancy
# temporal_skip = 1            # process every nth frame
# min_persistent_frames = 2    # number of consecutive frames a cell must be occupied to be considered static

# # ---- Load point clouds ----
# with open("cloud_time_history.pkl", "rb") as f:
#     point_clouds = pickle.load(f)

# print(f"Loaded {len(point_clouds)} scans")
# num_scans = len(point_clouds)

# # ---- Determine grid bounds ----
# all_points = np.vstack([pc for pc in point_clouds if pc.shape[0] > 0])
# x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
# y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

# nx = int((x_max - x_min) / res) + 1
# ny = int((y_max - y_min) / res) + 1

# print(f"Grid size: nx={nx}, ny={ny}")

# # ---- Initialize occupancy arrays ----
# prev_frame = np.zeros((ny, nx), dtype=bool)
# persistent_count = np.zeros((ny, nx), dtype=np.uint16)

# # ---- Iterate over scans ----
# for idx, pc in enumerate(point_clouds):
#     if idx % temporal_skip != 0:
#         continue

#     frame_occ = np.zeros((ny, nx), dtype=bool)

#     if pc.shape[0] > 0:
#         # Filter by z-range
#         pc = pc[(pc[:, 2] >= z_range[0]) & (pc[:, 2] <= z_range[1])]
#         x_idx = ((pc[:, 0] - x_min) / res).astype(int)
#         y_idx = ny - 1 - ((pc[:, 1] - y_min) / res).astype(int)
#         frame_occ[y_idx, x_idx] = True

#     # Temporal differencing: update persistent count
#     persistent_count[frame_occ] += 1
#     persistent_count[~frame_occ] = 0  # reset cells that are free

#     prev_frame = frame_occ

#     if (idx + 1) % 100 == 0:
#         print(f"Processed {idx+1}/{num_scans} frames")

# # ---- Generate static occupancy grid using persistence threshold ----
# grid_static = np.zeros((ny, nx), dtype=np.uint8)
# grid_static[persistent_count >= min_persistent_frames] = 0   # occupied
# grid_static[persistent_count < min_persistent_frames] = 255  # free

# # ---- Visualize ----
# plt.figure(figsize=(6, 6))
# plt.imshow(grid_static, cmap="gray", origin="upper")
# plt.title(f"Temporal Differencing: min_persistent_frames = {min_persistent_frames}")
# plt.axis("off")
# plt.show()

# # ---- Save result ----
# np.save("vanilla_occupancy_grid_temporal_diff.npy", grid_static)
# print("Saved vanilla occupancy grid using temporal differencing method")

# path_planning
Module for path planning and occupancy grid handling for the Unitree Go2 (Informed RRT* on a 2D occupancy grid from 3D point clouds).

## Layout

- **src/** (C++)
  - **occ_grid_bridge.hpp / .cpp** – Bridge between point clouds and occupancy grids: load map PCD (leveling, bounds), load edited PNG, world↔grid transform, pointcloud→occupancy, merge static+live, path indices→world coordinates.
  - **rrt_planner.hpp / .cpp** – Informed RRT* planner (from `irrt.cpp`): plans on a single-channel occupancy image (0=free, 255=obstacle), returns path as grid indices.
  - **path_planner_node.cpp** – ROS2 node: reads map PCD + edited PNG at init, subscribes to live PointCloud2, pose, and goal; runs RRT on overlaid grid; publishes path/waypoints in map frame and occupancy grid for visualization.

- **scripts/**
  - **visualize_waypoints_occ.py** – Matplotlib real-time view: subscribes to occupancy grid, planned path, waypoints, and robot pose; overlays them in map frame.

Topic names are placeholders; change them in the node and script to match your stack.

## Build (ROS2)

```bash
cd /path/to/path_planning
colcon build --packages-select path_planning
source install/setup.bash
```

## Run

1. Set params (e.g. in a launch file or `ros2 run`):
   - `map_pcd_path`, `map_png_path` – Paths to map PCD and edited PNG.
   - Optional: `resolution`, `z_min`, `z_max`, `robot_radius_px`, `rrt_iterations`, etc.

2. Start the path planner node (after starting localization and lidar):
   ```bash
   ros2 run path_planning path_planner_node --ros-args -p map_pcd_path:=/path/to/plab.pcd -p map_png_path:=/path/to/edited_map.png
   ```

3. Publish a goal (e.g. `geometry_msgs/PoseStamped` on `/goal_pose`).

4. Visualize:
   ```bash
   ros2 run path_planning visualize_waypoints_occ.py --ros-args --occ-topic /occupancy_grid --path-topic /planned_path --waypoints-topic /waypoints --pose-topic /pcl_pose
   ```
   Or run the script directly: `python3 scripts/visualize_waypoints_occ.py --rate 10`.

## Occupancy data modes and model integration

The node can run in three occupancy modes (`occ_data_mode`): **live**, **map_frame_model**, or **agent_centered_model**. In the two model modes, the node publishes input grids to a map-updater/model node and subscribes to predicted grids; planning then uses a **cookie-cutter** combination: static map everywhere except in the region of interest (ROI), where only the weighted predictions are used.

- **map_frame_model:** ROI is the axis-aligned rectangle given by `sample_col_min/max`, `sample_row_min/max`. The crop is centered on a 1216×1216 canvas for the model; predictions (1216×1216) are cookie-cut and pasted back into that rectangle on the full map.
- **agent_centered_model:** ROI is the ego-centered 201×201 subspace (≈10 m × 10 m) at the anchor pose. The map region covered by that ego footprint is zeroed, then the weighted predicted ego grid is pasted at the anchor using the same transform.

### Placing the predicted ego grid on the map (rotation)

The predicted agent-centered grid is in ego frame (robot-centered, robot-facing). To place it on the full map we do **not** rotate the occupancy image in discrete pixel space. Instead we use a **continuous world-space transform** and then snap to the map grid:

- For each ego-grid cell (row, col), we compute its position in ego world coordinates (meters) using the ego grid resolution and origin.
- We apply the 2D rotation by the anchor yaw and add the anchor position to get map-frame world coordinates.
- We convert (map_x, map_y) to map grid indices with `worldToGrid` and write the occupancy there.

So rotation is handled in meters; we never rotate the 201×201 image by an angle in pixel space (which would require interpolation or leave gaps). Each ego cell maps to one map cell; multiple ego cells can land in the same map cell, and some map cells in the rotated footprint may not be hit (forward mapping). This keeps the implementation simple and avoids sub-pixel artifacts.

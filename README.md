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

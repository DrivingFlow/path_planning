# path_planning
Module for path planning and occupancy grid handling for the Unitree Go2 (Informed RRT* / A* on a 2D occupancy grid from 3D point clouds).

## Layout

- **src/** (C++)
  - **occ_grid_bridge.hpp / .cpp** – Bridge between point clouds and occupancy grids: load map PCD (leveling, bounds), load edited PNG, world↔grid transform, pointcloud→occupancy, merge static+live, path indices→world coordinates.
  - **rrt_planner.hpp / .cpp** – Informed RRT* planner: plans on a single-channel occupancy image (0=free, 255=obstacle), returns path as grid indices.
  - **astar_energy_planner.hpp / .cpp** – A* planner with clearance-based energy cost: prefers high-clearance paths, includes elastic-band smoothing and arc-length resampling.
  - **path_planner_node.cpp** – ROS2 node: reads map PCD + edited PNG at init, subscribes to live PointCloud2, pose, and goal; runs RRT or A* on overlaid grid; publishes path/waypoints in map frame and occupancy grid for visualization.

- **scripts/**
  - **visualize_waypoints_occ.py** – Matplotlib real-time view: subscribes to occupancy grid, planned path, waypoints, and robot pose; overlays them in map frame.
  - **visualize_waypoints_occ.cpp** – C++ OpenCV real-time visualizer with side panel showing planner status, coordinates, zoom, and more.

## Build (ROS2)

```bash
cd /path/to/path_planning
colcon build --packages-select path_planning
source install/setup.bash
```

## Run

1. Set params (e.g. in the launch file or `ros2 run`):
   - `map_pcd_path`, `map_png_path` – Paths to map PCD and edited PNG.
   - Optional: `resolution`, `z_min`, `z_max`, `robot_radius`, etc.

2. Start the path planner node (after starting localization and lidar):
   ```bash
   ros2 run path_planning path_planner_node --ros-args -p map_pcd_path:=/path/to/plab.pcd -p map_png_path:=/path/to/edited_map.png
   ```

3. Publish a goal (e.g. `geometry_msgs/PoseStamped` on `/goal_pose`).

4. Visualize:
   ```bash
   ros2 launch path_planning path_planning.launch.py visualize:=True visualizer_type:=cpp
   ```

## Launch Parameters

### Map & Grid

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map_pcd_path` | string | *(see launch)* | Path to the PCD point cloud map file used to establish grid bounds and coordinate frame. |
| `map_png_path` | string | *(see launch)* | Path to the edited PNG occupancy grid image (0=free, 255=obstacle). Must match PCD grid dimensions. |
| `resolution` | double | `0.05` | Occupancy grid resolution in meters per cell (e.g. 0.05 = 5 cm/pixel). |
| `z_min` | double | `0.03` | Minimum z-coordinate (meters) for filtering 3D points into the 2D occupancy grid. Points below this are ignored. |
| `z_max` | double | `1.0` | Maximum z-coordinate (meters) for filtering 3D points. Points above this are ignored. |

### Robot Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robot_radius` | double | `0.15` | Robot radius in meters for obstacle inflation during collision checking. |
| `origin_crop_radius` | double | `0.06` | Radius (meters) of a circle around the robot to crop out self-observed obstacles (e.g. wifi adapter). 0 = disabled. |
| `origin_crop_forward_offset` | double | `0.20` | Forward offset (meters) of the crop circle center from the robot origin along +x (robot forward). |
| `crop_origin_in_model_input` | bool | `False` | In model modes, apply the same origin crop to the 201×201 frame before it is sent to the model. When false, origin cropping is only applied after fusion on the final combined map. |

### Planner Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planner` | string | `astar` | Path planner algorithm: `rrt` (Informed RRT*) or `astar` / `astar_energy` (A* with clearance energy cost). |
| `planner_settings` | string | `0.1,0.2,0.2,30` | A* only: comma-separated `beta_valley,smooth_alpha,smooth_beta,smooth_n_iter`. RRT ignores this. |
| `plan_interval_ms` | int | `100` | Planning cycle interval in milliseconds. In model modes, forced to 100 ms to match 10 Hz lidar. |

### RRT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rrt_iterations` | int | `10000` | Number of RRT* planning iterations. More = better path quality but slower. |
| `step_size` | double | `0.4` | RRT step size in meters: distance to extend the tree per iteration. Also used as default A* resample distance. |
| `rrt_goal_sample_rate` | double | `0.05` | Probability (0–1) that each RRT iteration samples the goal directly instead of a random point. |

### A* Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `astar_corridor_half_width` | double | `0.0` | Additional half-width (meters) to enforce a clear corridor around the A* centerline path. 0 = disabled. |

### Replanning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `replan_lookahead_distance` | double | `4.0` | Distance (meters) along path to check for obstacle intersection. If intersection is beyond this, don't trigger replan. |
| `replan_interval_sec` | double | `1.0` | Time (seconds) between periodic replans. Set to 0 to disable periodic replanning (only intersection/new-goal triggers). |
| `waypoint_spacing` | double | `0.4` | Arc-length spacing (meters) for resampling planner output into evenly spaced waypoints. Controls published path density. |
| `local_replan_enabled` | bool | `False` | If true, intersection-triggered replans only replan the path within `local_replan_radius` of the robot and splice with the original tail. |
| `local_replan_radius` | double | `5.0` | Radius (meters) for local replanning. The planner targets the existing path point at this distance from the robot. |

### Sampling Bounds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_col_min` | int | `420` | RRT sampling minimum column (grid index). -1 = use full grid width. |
| `sample_col_max` | int | `920` | RRT sampling maximum column (grid index). -1 = use full grid width. |
| `sample_row_min` | int | `60` | RRT sampling minimum row (grid index). -1 = use full grid height. |
| `sample_row_max` | int | `610` | RRT sampling maximum row (grid index). -1 = use full grid height. |
| `goal_in_pixels` | bool | `False` | If true, `/goal_pose` x,y are interpreted as grid col,row instead of world meters. |

### Lidar & Live Scans

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `live_scan_radius` | double | `0.0` | Maximum distance (meters) from robot for including live scan points. Points farther away are discarded. 0 = no limit (use all points). |

### Topics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lidar_topic` | string | `/livox/lidar` | Input topic for LIDAR point cloud (sensor_msgs/PointCloud2). |
| `pose_topic` | string | `/pcl_pose` | Input topic for robot pose (geometry_msgs/PoseWithCovarianceStamped). |
| `goal_topic` | string | `/move_base_simple/goal` | Input topic for goal pose (geometry_msgs/PoseStamped). |
| `planned_path_topic` | string | `/planned_path` | Output topic for the planned path (nav_msgs/Path). |
| `waypoints_topic` | string | `/waypoints` | Output topic for waypoints (nav_msgs/Path). |
| `occupancy_grid_topic` | string | `/occupancy_grid` | Output topic for the combined occupancy grid (nav_msgs/OccupancyGrid). |
| `planner_status_topic` | string | `/planner_status` | Output topic for planner status text (std_msgs/String). Consumed by the C++ visualizer side panel. |

### Model Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `occ_data_mode` | string | `live` | Occupancy data source: `live` (lidar only), `map_frame_model` (model predictions in map frame), or `agent_centered_model` (model predictions in ego frame). |
| `overlay_live_scans_with_model` | bool | `False` | In model modes, overlay live lidar obstacles onto model predictions for a conservative union of both obstacle sources. |
| `prediction_temperature` | double | `10000.0` | Boltzmann temperature for weighting predicted frames. Higher = more uniform weighting across frames. |
| `num_predicted_frames` | int | `5` | Number of predicted frames to use from model output. |
| `model_occupancy_threshold` | double | `0.3` | Threshold (0–1) for binarizing analog model output values before scaling to 0–255. |
| `model_occ_input_topic` | string | `/map_updater/occ_grid_input` | Topic for publishing occupancy grid input to the model node. |
| `model_predicted_output_topic` | string | `/map_updater/predicted_grid_output` | Topic for receiving predicted occupancy grids from the model node. |
| `agent_frame_stride` | int | `5` | Stride between frames sampled for the agent-centered model (queue size = 4×stride+1). |

### Visualizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `visualize` | bool | `False` | Enable visualization of path and occupancy grid (requires display / X forwarding). |
| `viz_rate` | double | `10.0` | Visualization update rate in Hz. |
| `visualizer_type` | string | `cpp` | Visualizer implementation: `cpp` (OpenCV-based) or `python` (Matplotlib-based). |
| `show_energy_map` | bool | `False` | Show clearance/energy map beside the binary occupancy map in the C++ visualizer. |
| `show_agent_centered_roi` | bool | `False` | Show agent-centered 5 m radius ROI panel in the C++ visualizer. |
| `show_robot_marker` | bool | `True` | Draw the green robot marker circle on the occupancy view. |
| `show_goal_marker` | bool | `True` | Draw the yellow goal star marker on the occupancy view. |
| `view_col_min` | int | `420` | Visualization view minimum column (grid index). -1 = full grid width. |
| `view_col_max` | int | `920` | Visualization view maximum column (grid index). -1 = full grid width. |
| `view_row_min` | int | `60` | Visualization view minimum row (grid index). -1 = full grid height. |
| `view_row_max` | int | `610` | Visualization view maximum row (grid index). -1 = full grid height. |

## Replanning Logic

The planner triggers a replan under the following conditions (in priority order):

1. **No current path** – always plans immediately.
2. **New goal received** – replans immediately when a new `/goal_pose` message arrives.
3. **Obstacle intersection** – replans immediately when the current path intersects obstacles within `replan_lookahead_distance`. If `local_replan_enabled` is true, only the portion of the path within `local_replan_radius` is replanned (the rest of the original path is preserved).
4. **Periodic timer** – replans every `replan_interval_sec` seconds if no other trigger fires.

If none of the above conditions are met, the planner publishes a "tracking/path_clear" status and skips the planning cycle.

## Occupancy Data Modes and Model Integration

The node can run in three occupancy modes (`occ_data_mode`): **live**, **map_frame_model**, or **agent_centered_model**. In the two model modes, the node publishes input grids to a map-updater/model node and subscribes to predicted grids; planning then uses a **cookie-cutter** combination: static map everywhere except in the region of interest (ROI), where only the weighted predictions are used.

- **map_frame_model:** ROI is the axis-aligned rectangle given by `sample_col_min/max`, `sample_row_min/max`. The crop is centered on a 1216×1216 canvas for the model; predictions (1216×1216) are cookie-cut and pasted back into that rectangle on the full map.
- **agent_centered_model:** ROI is the ego-centered 201×201 subspace (≈10 m × 10 m) at the anchor pose. The map region covered by that ego footprint is zeroed, then the weighted predicted ego grid is pasted at the anchor using the same transform.

### Placing the predicted ego grid on the map (rotation)

The predicted agent-centered grid is in ego frame (robot-centered, robot-facing). To place it on the full map we use a **continuous world-space transform** and then snap to the map grid:

- For each ego-grid cell (row, col), we compute its position in ego world coordinates (meters) using the ego grid resolution and origin.
- We apply the 2D rotation by the anchor yaw and add the anchor position to get map-frame world coordinates.
- We convert (map_x, map_y) to map grid indices with `worldToGrid` and write the occupancy there.

So rotation is handled in meters; we never rotate the 201×201 image by an angle in pixel space (which would require interpolation or leave gaps).

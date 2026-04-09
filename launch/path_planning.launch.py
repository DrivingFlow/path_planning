from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    map_pcd_path = LaunchConfiguration("map_pcd_path")
    map_png_path = LaunchConfiguration("map_png_path")
    resolution = LaunchConfiguration("resolution")
    z_min = LaunchConfiguration("z_min")
    z_max = LaunchConfiguration("z_max")
    robot_radius = LaunchConfiguration("robot_radius")
    origin_crop_radius = LaunchConfiguration("origin_crop_radius")
    origin_crop_forward_offset = LaunchConfiguration("origin_crop_forward_offset")
    origin_crop_lateral_offset = LaunchConfiguration("origin_crop_lateral_offset")
    crop_origin_in_model_input = LaunchConfiguration("crop_origin_in_model_input")
    astar_corridor_half_width = LaunchConfiguration("astar_corridor_half_width")
    use_model_overlay = LaunchConfiguration("use_model_overlay", default="True")
    astar_corridor_half_width_live = LaunchConfiguration("astar_corridor_half_width_live", default="0.20")
    rrt_iterations = LaunchConfiguration("rrt_iterations")
    step_size = LaunchConfiguration("step_size")
    rrt_goal_sample_rate = LaunchConfiguration("rrt_goal_sample_rate")
    replan_lookahead_distance = LaunchConfiguration("replan_lookahead_distance")
    replan_interval_sec = LaunchConfiguration("replan_interval_sec")
    waypoint_spacing = LaunchConfiguration("waypoint_spacing")
    local_replan_enabled = LaunchConfiguration("local_replan_enabled")
    local_replan_radius = LaunchConfiguration("local_replan_radius")
    live_scan_radius = LaunchConfiguration("live_scan_radius")
    min_replan_obstacle_size = LaunchConfiguration("min_replan_obstacle_size")
    door_toggle_enabled = LaunchConfiguration("door_toggle_enabled")
    door_toggle_maps = LaunchConfiguration("door_toggle_maps")

    lidar_topic = LaunchConfiguration("lidar_topic")
    pose_topic = LaunchConfiguration("pose_topic")
    goal_topic = LaunchConfiguration("goal_topic")
    planned_path_topic = LaunchConfiguration("planned_path_topic")
    waypoints_topic = LaunchConfiguration("waypoints_topic")
    occupancy_grid_topic = LaunchConfiguration("occupancy_grid_topic")
    planner_status_topic = LaunchConfiguration("planner_status_topic")

    visualize = LaunchConfiguration("visualize")
    viz_rate = LaunchConfiguration("viz_rate")
    visualizer_type = LaunchConfiguration("visualizer_type")
    view_col_min = LaunchConfiguration("view_col_min")
    view_col_max = LaunchConfiguration("view_col_max")
    view_row_min = LaunchConfiguration("view_row_min")
    view_row_max = LaunchConfiguration("view_row_max")
    show_energy_map = LaunchConfiguration("show_energy_map")
    show_agent_centered_roi = LaunchConfiguration("show_agent_centered_roi")
    show_robot_marker = LaunchConfiguration("show_robot_marker")
    show_goal_marker = LaunchConfiguration("show_goal_marker")

    sample_col_min = LaunchConfiguration("sample_col_min")
    sample_col_max = LaunchConfiguration("sample_col_max")
    sample_row_min = LaunchConfiguration("sample_row_min")
    sample_row_max = LaunchConfiguration("sample_row_max")
    goal_in_pixels = LaunchConfiguration("goal_in_pixels")
    planner = LaunchConfiguration("planner")
    planner_settings_str = LaunchConfiguration("planner_settings", default="0.1,0.1,0.2,50")

    occ_data_mode = LaunchConfiguration("occ_data_mode", default="live")
    overlay_live_scans_with_model = LaunchConfiguration("overlay_live_scans_with_model", default="False")
    prediction_temperature = LaunchConfiguration("prediction_temperature", default="3.0")
    num_predicted_frames = LaunchConfiguration("num_predicted_frames", default="5")
    model_occupancy_threshold = LaunchConfiguration("model_occupancy_threshold", default="0.5")
    model_occ_input_topic = LaunchConfiguration("model_occ_input_topic", default="/map_updater/occ_grid_input")
    model_predicted_output_topic = LaunchConfiguration("model_predicted_output_topic", default="/map_updater/predicted_grid_output")
    agent_frame_stride = LaunchConfiguration("agent_frame_stride", default="5")
    max_prediction_age_ms = LaunchConfiguration("max_prediction_age_ms", default="250.0")
    use_in_process_model = LaunchConfiguration("use_in_process_model", default="False")
    model_script_path = LaunchConfiguration("model_script_path",
        default="/home/unitree/path_planning/src/path_planning/models/model_scripted.pt")

    return LaunchDescription(
        [
            # Path to the PCD map file
            DeclareLaunchArgument("map_pcd_path", default_value="/home/unitree/path_planning/src/path_planning/utils/alumni_final_1_rotated.pcd"),
            # Path to the PNG occupancy grid map file
            DeclareLaunchArgument("map_png_path", default_value="/home/unitree/path_planning/src/path_planning/utils/alumni_final_1_rotated_toggle1.png"),
            # Resolution of the occupancy grid (meters per cell)
            DeclareLaunchArgument("resolution", default_value="0.05"),
            # Minimum z-coordinate for obstacle detection (meters)
            DeclareLaunchArgument("z_min", default_value="0.03"),
            # Maximum z-coordinate for obstacle detection (meters)
            DeclareLaunchArgument("z_max", default_value="1.0"),
            # Robot radius for collision checking (meters)
            DeclareLaunchArgument("robot_radius", default_value="0.15"),
            # Radius (meters) of circle to crop out obstacles (e.g. wifi adapter). 0 = disabled.
            DeclareLaunchArgument("origin_crop_radius", default_value="0.20"),
            # Forward offset (meters) of crop circle center from robot origin along +x.
            DeclareLaunchArgument("origin_crop_forward_offset", default_value="0.17"),
            # Lateral offset (meters) of crop circle center from robot origin along +y (robot left).
            DeclareLaunchArgument("origin_crop_lateral_offset", default_value="0.05"),
            # If true, apply the origin crop to model input frames before inference; if false, crop only after fusion.
            DeclareLaunchArgument("crop_origin_in_model_input", default_value="False"),
            # A* only: additional corridor half-width around centerline (meters)
            DeclareLaunchArgument("astar_corridor_half_width", default_value="0.0"),
            # Toggle model prediction overlay on/off at runtime (only relevant in model modes)
            DeclareLaunchArgument("use_model_overlay", default_value="True"),
            # Live-adjustable A* corridor half-width (meters); changed via visualizer W/S keys
            DeclareLaunchArgument("astar_corridor_half_width_live", default_value="0.20"),
            # Number of RRT iterations for path planning
            DeclareLaunchArgument("rrt_iterations", default_value="10000"),
            # Step size for RRT tree expansion (meters)
            DeclareLaunchArgument("step_size", default_value="0.4"),
            # Probability of sampling the goal during RRT planning
            DeclareLaunchArgument("rrt_goal_sample_rate", default_value="0.05"),
            # Lookahead distance for replanning (meters)
            DeclareLaunchArgument("replan_lookahead_distance", default_value="4.0"),
            # Time interval between periodic replanning (seconds); 0 = disable periodic replan
            DeclareLaunchArgument("replan_interval_sec", default_value="10.0"),
            # Waypoint spacing (meters) for arc-length resampling of planner output
            DeclareLaunchArgument("waypoint_spacing", default_value="0.4"),
            # Enable local replanning: intersection-triggered replans only affect path within local_replan_radius
            DeclareLaunchArgument("local_replan_enabled", default_value="False"),
            # Radius (meters) for local replanning around robot position
            DeclareLaunchArgument("local_replan_radius", default_value="5.0"),
            # Maximum distance (meters) from robot for live scan points; 0 = no limit (use all points)
            DeclareLaunchArgument("live_scan_radius", default_value="5.0"),
            # Minimum obstacle blob size (pixels) to trigger intersection-based replanning; smaller blobs are ignored. 0 = disabled.
            DeclareLaunchArgument("min_replan_obstacle_size", default_value="0"),
            # Enable door toggle: press 'o' in visualizer to swap between two static map PNGs
            DeclareLaunchArgument("door_toggle_enabled", default_value="True"),
            # Comma-separated pair of PNG paths for door toggle (map1,map2)
            DeclareLaunchArgument("door_toggle_maps", default_value="/home/unitree/path_planning/src/path_planning/utils/alumni_final_1_rotated_toggle1.png,/home/unitree/path_planning/src/path_planning/utils/alumni_final_1_rotated_toggle2.png"),
            # Topic name for LIDAR point cloud input
            DeclareLaunchArgument("lidar_topic", default_value="/livox/lidar"),
            # Topic name for robot pose input
            DeclareLaunchArgument("pose_topic", default_value="/pcl_pose"),
            # Topic name for goal pose input
            DeclareLaunchArgument("goal_topic", default_value="/move_base_simple/goal"),
            # Topic name for publishing planned path
            DeclareLaunchArgument("planned_path_topic", default_value="/planned_path"),
            # Topic name for publishing waypoints
            DeclareLaunchArgument("waypoints_topic", default_value="/waypoints"),
            # Topic name for publishing occupancy grid
            DeclareLaunchArgument("occupancy_grid_topic", default_value="/occupancy_grid"),
            # Topic name for publishing planner status text for visualization
            DeclareLaunchArgument("planner_status_topic", default_value="/planner_status"),
            # Enable visualization of path and occupancy grid (requires X display)
            DeclareLaunchArgument("visualize", default_value="True"),
            # Visualization update rate (Hz)
            DeclareLaunchArgument("viz_rate", default_value="10.0"),
            # Visualizer type: 'cpp' or 'python'
            DeclareLaunchArgument("visualizer_type", default_value="cpp"),
            # Whether to show energy / clearance map beside binary map in C++ visualizer
            DeclareLaunchArgument("show_energy_map", default_value="False"),
            # Whether to show agent-centered 5 m radius ROI panel in C++ visualizer (independent of energy map)
            DeclareLaunchArgument("show_agent_centered_roi", default_value="False"),
            # Whether to draw the green robot marker on the occupancy view
            DeclareLaunchArgument("show_robot_marker", default_value="True"),
            # Whether to draw the yellow goal marker on the occupancy view
            DeclareLaunchArgument("show_goal_marker", default_value="True"),
            # Minimum column for visualization view; -1 = full grid width
            DeclareLaunchArgument("view_col_min", default_value="420"),
            # Maximum column for visualization view; -1 = full grid width
            DeclareLaunchArgument("view_col_max", default_value="920"),
            # Minimum row for visualization view; -1 = full grid height
            DeclareLaunchArgument("view_row_min", default_value="60"),
            # Maximum row for visualization view; -1 = full grid height
            DeclareLaunchArgument("view_row_max", default_value="610"),
            # RRT sampling min col (grid index); -1 = full grid
            DeclareLaunchArgument("sample_col_min", default_value="420"),
            # RRT sampling max col (grid index); -1 = full grid
            DeclareLaunchArgument("sample_col_max", default_value="920"),
            # RRT sampling min row (grid index); -1 = full grid
            DeclareLaunchArgument("sample_row_min", default_value="60"),
            # RRT sampling max row (grid index); -1 = full grid
            DeclareLaunchArgument("sample_row_max", default_value="610"),
            # If True, /goal_pose x,y are grid col,row; else world (m)
            DeclareLaunchArgument("goal_in_pixels", default_value="False"),
            # Path planner: 'rrt' or 'astar' (A* with clearance energy)
            DeclareLaunchArgument("planner", default_value="astar"),
            # A* only: [beta_valley,smooth_alpha,smooth_beta,smooth_n_iter]; RRT ignores
            DeclareLaunchArgument("planner_settings", default_value="0.1,0.2,0.2,30"),
            # Occupancy data source: 'live' | 'map_frame_model' | 'agent_centered_model'
            DeclareLaunchArgument("occ_data_mode", default_value="live"),
            # In model modes, overlay live lidar obstacles onto model predictions for a conservative union.
            DeclareLaunchArgument("overlay_live_scans_with_model", default_value="False"),
            DeclareLaunchArgument("prediction_temperature", default_value="10000.0"),
            DeclareLaunchArgument("num_predicted_frames", default_value="5"),
            # Threshold for binarizing analog model output values (0-1 range) before scaling to 0-255
            DeclareLaunchArgument("model_occupancy_threshold", default_value="0.3"),
            DeclareLaunchArgument("model_occ_input_topic", default_value="/map_updater/occ_grid_input"),
            DeclareLaunchArgument("model_predicted_output_topic", default_value="/map_updater/predicted_grid_output"),
            # Stride between frames sampled for the agent-centered model (queue size = 4*stride+1)
            DeclareLaunchArgument("agent_frame_stride", default_value="5"),
            DeclareLaunchArgument("max_prediction_age_ms", default_value="250.0"),
            # Run the ML model in-process via TorchScript instead of via external ROS node
            DeclareLaunchArgument("use_in_process_model", default_value="False"),
            # Path to the exported TorchScript model file (.pt)
            DeclareLaunchArgument("model_script_path",
                default_value="/home/unitree/path_planning/src/path_planning/models/model_scripted.pt"),
            Node(
                package="path_planning",
                executable="path_planner_node",
                name="path_planner",
                output="screen",
                parameters=[
                    {
                        "map_pcd_path": map_pcd_path,
                        "map_png_path": map_png_path,
                        "resolution": resolution,
                        "z_min": z_min,
                        "z_max": z_max,
                        "robot_radius": robot_radius,
                        "origin_crop_radius": origin_crop_radius,
                        "origin_crop_forward_offset": origin_crop_forward_offset,
                        "origin_crop_lateral_offset": origin_crop_lateral_offset,
                        "crop_origin_in_model_input": crop_origin_in_model_input,
                        "astar_corridor_half_width": astar_corridor_half_width,
                        "use_model_overlay": use_model_overlay,
                        "astar_corridor_half_width_live": astar_corridor_half_width_live,
                        "rrt_iterations": rrt_iterations,
                        "step_size": step_size,
                        "rrt_goal_sample_rate": rrt_goal_sample_rate,
                        "replan_lookahead_distance": replan_lookahead_distance,
                        "replan_interval_sec": replan_interval_sec,
                        "waypoint_spacing": waypoint_spacing,
                        "local_replan_enabled": local_replan_enabled,
                        "local_replan_radius": local_replan_radius,
                        "live_scan_radius": live_scan_radius,
                        "min_replan_obstacle_size": min_replan_obstacle_size,
                        "door_toggle_enabled": door_toggle_enabled,
                        "door_toggle_maps": door_toggle_maps,
                        "sample_col_min": sample_col_min,
                        "sample_col_max": sample_col_max,
                        "sample_row_min": sample_row_min,
                        "sample_row_max": sample_row_max,
                        "goal_in_pixels": goal_in_pixels,
                        "planner": planner,
                        "planner_settings": planner_settings_str,
                        "planner_status_topic": planner_status_topic,
                        "occ_data_mode": occ_data_mode,
                        "overlay_live_scans_with_model": overlay_live_scans_with_model,
                        "prediction_temperature": prediction_temperature,
                        "num_predicted_frames": num_predicted_frames,
                        "model_occupancy_threshold": model_occupancy_threshold,
                        "model_occ_input_topic": model_occ_input_topic,
                        "model_predicted_output_topic": model_predicted_output_topic,
                        "agent_frame_stride": agent_frame_stride,
                        "max_prediction_age_ms": max_prediction_age_ms,
                        "use_in_process_model": use_in_process_model,
                        "model_script_path": model_script_path,
                    }
                ],
                remappings=[
                    ("/lidar_map", lidar_topic),
                    ("/pcl_pose", pose_topic),
                    ("/goal_pose", goal_topic),
                    ("/planned_path", planned_path_topic),
                    ("/waypoints", waypoints_topic),
                    ("/occupancy_grid", occupancy_grid_topic),
                ],
            ),
            Node(
                package="path_planning",
                executable="visualize_waypoints_occ.py",
                name="waypoints_occ_visualizer",
                output="screen",
                arguments=[
                    "--occ-topic",
                    occupancy_grid_topic,
                    "--path-topic",
                    planned_path_topic,
                    "--waypoints-topic",
                    waypoints_topic,
                    "--pose-topic",
                    pose_topic,
                    "--goal-topic",
                    goal_topic,
                    "--rate",
                    viz_rate,
                    "--view-col-min",
                    view_col_min,
                    "--view-col-max",
                    view_col_max,
                    "--view-row-min",
                    view_row_min,
                    "--view-row-max",
                    view_row_max,
                ],
                condition=IfCondition(
                    PythonExpression([visualize, " and '", visualizer_type, "' == 'python'"])
                ),
            ),
            Node(
                package="path_planning",
                executable="visualize_waypoints_occ_cpp",
                name="waypoints_occ_visualizer",
                output="screen",
                parameters=[
                    {
                        "occ_topic": occupancy_grid_topic,
                        "path_topic": planned_path_topic,
                        "waypoints_topic": waypoints_topic,
                        "pose_topic": pose_topic,
                        "goal_topic": goal_topic,
                        "planner_status_topic": planner_status_topic,
                        "rate": viz_rate,
                        "view_col_min": view_col_min,
                        "view_col_max": view_col_max,
                        "view_row_min": view_row_min,
                        "view_row_max": view_row_max,
                        "show_energy_map": show_energy_map,
                        "show_agent_centered_roi": show_agent_centered_roi,
                        "show_robot_marker": show_robot_marker,
                        "show_goal_marker": show_goal_marker,
                        "door_toggle_enabled": door_toggle_enabled,
                    }
                ],
                condition=IfCondition(
                    PythonExpression([visualize, " and '", visualizer_type, "' == 'cpp'"])
                ),
            ),
        ]
    )

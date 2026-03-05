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
    rrt_iterations = LaunchConfiguration("rrt_iterations")
    step_size = LaunchConfiguration("step_size")
    rrt_goal_sample_rate = LaunchConfiguration("rrt_goal_sample_rate")
    replan_lookahead_distance = LaunchConfiguration("replan_lookahead_distance")
    replan_interval_sec = LaunchConfiguration("replan_interval_sec")

    lidar_topic = LaunchConfiguration("lidar_topic")
    pose_topic = LaunchConfiguration("pose_topic")
    goal_topic = LaunchConfiguration("goal_topic")
    planned_path_topic = LaunchConfiguration("planned_path_topic")
    waypoints_topic = LaunchConfiguration("waypoints_topic")
    occupancy_grid_topic = LaunchConfiguration("occupancy_grid_topic")

    visualize = LaunchConfiguration("visualize")
    viz_rate = LaunchConfiguration("viz_rate")
    visualizer_type = LaunchConfiguration("visualizer_type")
    view_col_min = LaunchConfiguration("view_col_min")
    view_col_max = LaunchConfiguration("view_col_max")
    view_row_min = LaunchConfiguration("view_row_min")
    view_row_max = LaunchConfiguration("view_row_max")
    show_energy_map = LaunchConfiguration("show_energy_map")

    sample_col_min = LaunchConfiguration("sample_col_min")
    sample_col_max = LaunchConfiguration("sample_col_max")
    sample_row_min = LaunchConfiguration("sample_row_min")
    sample_row_max = LaunchConfiguration("sample_row_max")
    goal_in_pixels = LaunchConfiguration("goal_in_pixels")
    planner = LaunchConfiguration("planner")
    planner_settings_str = LaunchConfiguration("planner_settings", default="0.1,0.1,0.2,50")

    return LaunchDescription(
        [
            # Path to the PCD map file
            DeclareLaunchArgument("map_pcd_path", default_value="/home/dog22/path_planning_ws/src/path_planning/utils/plab_4-1_rotated.pcd"),
            # Path to the PNG occupancy grid map file
            DeclareLaunchArgument("map_png_path", default_value="/home/dog22/path_planning_ws/src/path_planning/utils/plab_4-1_rotated.png"),
            # Resolution of the occupancy grid (meters per cell)
            DeclareLaunchArgument("resolution", default_value="0.05"),
            # Minimum z-coordinate for obstacle detection (meters)
            DeclareLaunchArgument("z_min", default_value="0.015"),
            # Maximum z-coordinate for obstacle detection (meters)
            DeclareLaunchArgument("z_max", default_value="0.6"),
            # Robot radius for collision checking (meters)
            DeclareLaunchArgument("robot_radius", default_value="0.1"),
            # Number of RRT iterations for path planning
            DeclareLaunchArgument("rrt_iterations", default_value="10000"),
            # Step size for RRT tree expansion (meters)
            DeclareLaunchArgument("step_size", default_value="0.4"),
            # Probability of sampling the goal during RRT planning
            DeclareLaunchArgument("rrt_goal_sample_rate", default_value="0.05"),
            # Lookahead distance for replanning (meters)
            DeclareLaunchArgument("replan_lookahead_distance", default_value="4.0"),
            # Time interval between replanning updates (seconds)
            DeclareLaunchArgument("replan_interval_sec", default_value="1.0"),
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
            # Enable visualization of path and occupancy grid
            DeclareLaunchArgument("visualize", default_value="True"),
            # Visualization update rate (Hz)
            DeclareLaunchArgument("viz_rate", default_value="10.0"),
            # Visualizer type: 'cpp' or 'python'
            DeclareLaunchArgument("visualizer_type", default_value="cpp"),
            # Whether to show energy / clearance map beside binary map in C++ visualizer
            DeclareLaunchArgument("show_energy_map", default_value="True"),
            # Minimum column for visualization view; -1 = full grid width
            DeclareLaunchArgument("view_col_min", default_value="375"),
            # Maximum column for visualization view; -1 = full grid width
            DeclareLaunchArgument("view_col_max", default_value="800"),
            # Minimum row for visualization view; -1 = full grid height
            DeclareLaunchArgument("view_row_min", default_value="75"),
            # Maximum row for visualization view; -1 = full grid height
            DeclareLaunchArgument("view_row_max", default_value="730"),
            # RRT sampling min col (grid index); -1 = full grid
            DeclareLaunchArgument("sample_col_min", default_value="375"),
            # RRT sampling max col (grid index); -1 = full grid
            DeclareLaunchArgument("sample_col_max", default_value="800"),
            # RRT sampling min row (grid index); -1 = full grid
            DeclareLaunchArgument("sample_row_min", default_value="75"),
            # RRT sampling max row (grid index); -1 = full grid
            DeclareLaunchArgument("sample_row_max", default_value="730"),
            # If True, /goal_pose x,y are grid col,row; else world (m)
            DeclareLaunchArgument("goal_in_pixels", default_value="False"),
            # Path planner: 'rrt' or 'astar' (A* with clearance energy)
            DeclareLaunchArgument("planner", default_value="astar"),
            # A* only: [beta_valley,smooth_alpha,smooth_beta,smooth_n_iter]; RRT ignores
            DeclareLaunchArgument("planner_settings", default_value="0.1,0.2,0.2,30"),
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
                        "rrt_iterations": rrt_iterations,
                        "step_size": step_size,
                        "rrt_goal_sample_rate": rrt_goal_sample_rate,
                        "replan_lookahead_distance": replan_lookahead_distance,
                        "replan_interval_sec": replan_interval_sec,
                        "sample_col_min": sample_col_min,
                        "sample_col_max": sample_col_max,
                        "sample_row_min": sample_row_min,
                        "sample_row_max": sample_row_max,
                        "goal_in_pixels": goal_in_pixels,
                        "planner": planner,
                        "planner_settings": planner_settings_str,
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
                        "rate": viz_rate,
                        "view_col_min": view_col_min,
                        "view_col_max": view_col_max,
                        "view_row_min": view_row_min,
                        "view_row_max": view_row_max,
                        "show_energy_map": show_energy_map,
                    }
                ],
                condition=IfCondition(
                    PythonExpression([visualize, " and '", visualizer_type, "' == 'cpp'"])
                ),
            ),
        ]
    )

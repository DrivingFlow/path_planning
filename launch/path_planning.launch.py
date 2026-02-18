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
    rrt_step_size = LaunchConfiguration("rrt_step_size")
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

    sample_col_min = LaunchConfiguration("sample_col_min")
    sample_col_max = LaunchConfiguration("sample_col_max")
    sample_row_min = LaunchConfiguration("sample_row_min")
    sample_row_max = LaunchConfiguration("sample_row_max")
    goal_in_pixels = LaunchConfiguration("goal_in_pixels")
    planner = LaunchConfiguration("planner")
    planner_settings_str = LaunchConfiguration("planner_settings", default="0.1,0.1,0.2,50")

    return LaunchDescription(
        [
            DeclareLaunchArgument("map_pcd_path", default_value="/home/dog22/path_planning_ws/src/path_planning/utils/plab_4-1_rotated.pcd"),
            DeclareLaunchArgument("map_png_path", default_value="/home/dog22/path_planning_ws/src/path_planning/utils/plab_4-1_rotated.png"),
            DeclareLaunchArgument("resolution", default_value="0.05"),
            DeclareLaunchArgument("z_min", default_value="0.1"),
            DeclareLaunchArgument("z_max", default_value="2.0"),
            DeclareLaunchArgument("robot_radius", default_value="0.0"),
            DeclareLaunchArgument("rrt_iterations", default_value="10000"),
            DeclareLaunchArgument("rrt_step_size", default_value="0.4"),
            DeclareLaunchArgument("rrt_goal_sample_rate", default_value="0.05"),
            DeclareLaunchArgument("replan_lookahead_distance", default_value="4.0"),
            DeclareLaunchArgument("replan_interval_sec", default_value="1.0"),
            DeclareLaunchArgument("lidar_topic", default_value="/livox/lidar"),
            DeclareLaunchArgument("pose_topic", default_value="/pcl_pose"),
            DeclareLaunchArgument("goal_topic", default_value="/move_base_simple/goal"),
            DeclareLaunchArgument("planned_path_topic", default_value="/planned_path"),
            DeclareLaunchArgument("waypoints_topic", default_value="/waypoints"),
            DeclareLaunchArgument("occupancy_grid_topic", default_value="/occupancy_grid"),
            DeclareLaunchArgument("visualize", default_value="True"),
            DeclareLaunchArgument("viz_rate", default_value="10.0"),
            DeclareLaunchArgument("visualizer_type", default_value="cpp"),
            DeclareLaunchArgument("view_col_min", default_value="-1"),
            DeclareLaunchArgument("view_col_max", default_value="-1"),
            DeclareLaunchArgument("view_row_min", default_value="-1"),
            DeclareLaunchArgument("view_row_max", default_value="-1"),
            DeclareLaunchArgument("sample_col_min", default_value="-1",
                description="RRT sampling min col (grid index); -1 = full grid"),
            DeclareLaunchArgument("sample_col_max", default_value="-1",
                description="RRT sampling max col (grid index); -1 = full grid"),
            DeclareLaunchArgument("sample_row_min", default_value="-1",
                description="RRT sampling min row (grid index); -1 = full grid"),
            DeclareLaunchArgument("sample_row_max", default_value="-1",
                description="RRT sampling max row (grid index); -1 = full grid"),
            DeclareLaunchArgument("goal_in_pixels", default_value="False",
                description="If True, /goal_pose x,y are grid col,row; else world (m)"),
            DeclareLaunchArgument("planner", default_value="rrt",
                description="Path planner: 'rrt' or 'astar' (A* with clearance energy)"),
            DeclareLaunchArgument("planner_settings", default_value="0.1,0.1,0.2,50",
                description="A* only: [beta_valley,smooth_alpha,smooth_beta,smooth_n_iter]; RRT ignores"),
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
                        "rrt_step_size": rrt_step_size,
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
                    }
                ],
                condition=IfCondition(
                    PythonExpression([visualize, " and '", visualizer_type, "' == 'cpp'"])
                ),
            ),
        ]
    )

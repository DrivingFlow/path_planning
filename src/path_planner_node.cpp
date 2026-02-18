/**
 * ROS2 path planner node for Unitree Go2.
 * Reads map PCD + edited PNG, subscribes to live lidar scans and goal,
 * runs Informed RRT* on the overlaid occupancy grid, converts path to
 * map-frame coordinates and publishes waypoints for the controller.
 *
 * Topic names are placeholders; adjust to match your stack.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/header.hpp>

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <mutex>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "occ_grid_bridge.hpp"
#include "rrt_planner.hpp"
#include "astar_energy_planner.hpp"

// Placeholder: replace with your waypoint message type if different (e.g. nav_msgs/Path or custom)
using WaypointArray = std::vector<std::array<double, 2>>;

class PathPlannerNode : public rclcpp::Node {
public:
    PathPlannerNode() : Node("path_planner") {
        declare_parameter<std::string>("map_pcd_path", "");
        declare_parameter<std::string>("map_png_path", "");
        // Grid resolution in meters (e.g., 0.05 = 5cm per pixel)
        declare_parameter<double>("resolution", 0.05);
        // Height range in meters for occupancy grid (points outside this range are ignored)
        declare_parameter<double>("z_min", 0.1);
        declare_parameter<double>("z_max", 2.0);
        // Robot radius in meters (converted to pixels internally for obstacle inflation)
        declare_parameter<double>("robot_radius", 0.25);
        // Number of RRT* planning iterations (more = better path but slower)
        declare_parameter<int>("rrt_iterations", 10000);
        // RRT* step size in meters (distance to extend tree per iteration, converted to pixels internally)
        declare_parameter<double>("rrt_step_size", 0.4);
        // Goal sample rate (probability 0-1): fraction of iterations that sample the goal directly
        // Higher values (e.g., 0.1) bias toward goal, lower (e.g., 0.01) explore more randomly
        declare_parameter<double>("rrt_goal_sample_rate", 0.05);
        // Lookahead distance in meters: if obstacle intersection is beyond this distance along path, don't replan
        // (assumes moving objects will be gone by the time robot reaches that point)
        declare_parameter<double>("replan_lookahead_distance", 4.0);
        // Force replan every N seconds (e.g. when manually driving so path drifts; 0 = disable periodic replan)
        declare_parameter<double>("replan_interval_sec", 5.0);
        // RRT sampling bounds (grid indices); -1 = use full grid
        declare_parameter<int>("sample_col_min", -1);
        declare_parameter<int>("sample_col_max", -1);
        declare_parameter<int>("sample_row_min", -1);
        declare_parameter<int>("sample_row_max", -1);
        // If true, /goal_pose x,y are interpreted as grid col,row; else as world (m)
        declare_parameter<bool>("goal_in_pixels", false);
        // Planner algorithm: "rrt" or "astar" (A* with clearance energy)
        declare_parameter<std::string>("planner", "rrt");
        // A* only: [beta_valley, smooth_alpha, smooth_beta, smooth_n_iter]; RRT ignores this (string or double array)
        declare_parameter("planner_settings", std::string("0.1,0.1,0.2,50"));

        std::string map_pcd = get_parameter("map_pcd_path").as_string();
        std::string map_png = get_parameter("map_png_path").as_string();
        if (map_pcd.empty() || map_png.empty()) {
            RCLCPP_WARN(get_logger(), "map_pcd_path or map_png_path not set; bridge will not be ready.");
        } else {
            initBridge(map_pcd, map_png);
        }

        // Placeholder topic names
        sub_live_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_map", 10, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                onPointCloud(msg);
            });
        sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/pcl_pose", 10, [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                onPose(msg);
            });
        sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                onGoal(msg);
            });

        pub_path_ = create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
        pub_waypoints_ = create_publisher<nav_msgs::msg::Path>("/waypoints", 10);
        pub_occ_grid_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_grid", 10);

        plan_timer_ = create_wall_timer(
            std::chrono::milliseconds(500),
            [this]() { runPlanningCycle(); });
    }

private:
    void initBridge(const std::string& map_pcd, const std::string& map_png) {
        double res = get_parameter("resolution").as_double();
        if (!bridge_.loadMapPcd(map_pcd, res)) {
            RCLCPP_ERROR(get_logger(), "Failed to load map PCD: %s", map_pcd.c_str());
            return;
        }
        RCLCPP_INFO(get_logger(),
            "Map PCD bounds: x=[%.4f, %.4f], y=[%.4f, %.4f], res=%.4f -> grid %d x %d",
            bridge_.xMin(), bridge_.xMax(), bridge_.yMin(), bridge_.yMax(),
            res, bridge_.gridWidth(), bridge_.gridHeight());
        int pcd_grid_w = bridge_.gridWidth();
        int pcd_grid_h = bridge_.gridHeight();
        if (!bridge_.loadEditedMapPng(map_png)) {
            RCLCPP_ERROR(get_logger(), "Failed to load map PNG: %s", map_png.c_str());
            return;
        }
        if (bridge_.gridWidth() != pcd_grid_w || bridge_.gridHeight() != pcd_grid_h) {
            RCLCPP_WARN(get_logger(),
                "Dimension mismatch: map PCD grid is %d x %d but PNG is %d x %d. Using PNG dimensions. "
                "Ensure the PNG was generated from the same PCD with the same resolution.",
                pcd_grid_w, pcd_grid_h, bridge_.gridWidth(), bridge_.gridHeight());
        }
        RCLCPP_INFO(get_logger(), "Bridge initialized: grid %d x %d", bridge_.gridWidth(), bridge_.gridHeight());
        RCLCPP_INFO(get_logger(), "Bridge w_/h_ (gridWidth/gridHeight): %d x %d",
            bridge_.gridWidth(), bridge_.gridHeight());
    }

    void onPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        live_points_.clear();
        int point_step = msg->point_step;
        int num = msg->width * msg->height;
        int x_off = -1, y_off = -1, z_off = -1;
        for (const auto& f : msg->fields) {
            if (f.name == "x") x_off = f.offset;
            if (f.name == "y") y_off = f.offset;
            if (f.name == "z") z_off = f.offset;
        }
        if (x_off < 0 || y_off < 0 || z_off < 0) return;
        const uint8_t* base = msg->data.data();
        for (int i = 0; i < num; ++i) {
            const uint8_t* pt = base + i * point_step;
            float x = *reinterpret_cast<const float*>(pt + x_off);
            float y = *reinterpret_cast<const float*>(pt + y_off);
            float z = *reinterpret_cast<const float*>(pt + z_off);
            live_points_.push_back({{x, y, z}});
        }
    }

    void onPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        start_x_ = msg->pose.pose.position.x;
        start_y_ = msg->pose.pose.position.y;
        start_z_ = msg->pose.pose.position.z;

        const auto& q = msg->pose.pose.orientation;
        tf2::Quaternion quat(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        start_yaw_ = yaw;
        have_pose_ = true;
    }

    void onGoal(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        goal_x_ = msg->pose.position.x;
        goal_y_ = msg->pose.position.y;
        have_goal_ = true;
        RCLCPP_INFO(get_logger(), "Received goal pose: x=%.3f, y=%.3f", goal_x_, goal_y_);
    }

    /**
     * Check if current path intersects obstacles in the occupancy grid.
     * Returns true if intersection found within lookahead_distance along path from robot position.
     * Path is in world coordinates, grid is occupancy grid (0=free, 255=obstacle).
     */
    bool pathIntersectsObstacles(const std::vector<std::array<double, 2>>& path_world,
                                 double robot_x, double robot_y,
                                 const cv::Mat& occupancy_grid,
                                 double lookahead_distance) const {
        if (path_world.empty()) return false;

        double resolution = bridge_.resolution();
        double cumulative_dist = 0.0;

        // Find closest point on path to robot (start checking from there)
        size_t start_idx = 0;
        double min_dist_sq = 1e30;
        for (size_t i = 0; i < path_world.size(); ++i) {
            double dx = path_world[i][0] - robot_x;
            double dy = path_world[i][1] - robot_y;
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                start_idx = i;
            }
        }

        // Check path segments starting from closest point
        for (size_t i = start_idx; i + 1 < path_world.size(); ++i) {
            double x0 = path_world[i][0], y0 = path_world[i][1];
            double x1 = path_world[i+1][0], y1 = path_world[i+1][1];

            // Segment length
            double seg_len = std::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
            if (i > start_idx) {
                cumulative_dist += seg_len;
            } else {
                // For first segment, add distance from robot to segment start
                double dx = x0 - robot_x, dy = y0 - robot_y;
                cumulative_dist = std::sqrt(dx*dx + dy*dy);
            }

            // Stop if we've exceeded lookahead distance
            if (cumulative_dist > lookahead_distance) {
                break;
            }

            // Convert segment endpoints to grid coordinates
            int col0, row0, col1, row1;
            bridge_.worldToGrid(x0, y0, col0, row0);
            bridge_.worldToGrid(x1, y1, col1, row1);

            // Sample points along segment and check for obstacles
            int num_samples = static_cast<int>(seg_len / resolution) + 1;
            for (int j = 0; j <= num_samples; ++j) {
                double t = static_cast<double>(j) / num_samples;
                int col = static_cast<int>(col0 + t * (col1 - col0));
                int row = static_cast<int>(row0 + t * (row1 - row0));

                if (col >= 0 && col < occupancy_grid.cols && row >= 0 && row < occupancy_grid.rows) {
                    if (occupancy_grid.at<uchar>(row, col) != 0) {
                        // Obstacle found within lookahead distance
                        return true;
                    }
                }
            }
        }

        return false;
    }

    std::vector<std::array<float, 3>> transformPointsToMap(
        const std::vector<std::array<float, 3>>& points_lidar,
        double tx, double ty, double tz, double yaw) const {
        if (points_lidar.empty()) return {};
        std::vector<std::array<float, 3>> out;
        out.reserve(points_lidar.size());
        double c = std::cos(yaw);
        double s = std::sin(yaw);
        for (const auto& pt : points_lidar) {
            double x = pt[0];
            double y = pt[1];
            double z = pt[2];
            double x_m = c * x - s * y + tx;
            double y_m = s * x + c * y + ty;
            double z_m = z + tz;
            out.push_back({{static_cast<float>(x_m), static_cast<float>(y_m), static_cast<float>(z_m)}});
        }
        return out;
    }

    void runPlanningCycle() {
        if (!bridge_.isInitialized()) return;

        std::vector<std::array<float, 3>> live_pts;
        double start_x, start_y, goal_x, goal_y;
        double start_z, start_yaw;
        bool have_pose, have_goal;
        std::vector<std::array<double, 2>> current_path;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            live_pts = live_points_;
            start_x = start_x_; start_y = start_y_;
            start_z = start_z_; start_yaw = start_yaw_;
            goal_x = goal_x_; goal_y = goal_y_;
            have_pose = have_pose_; have_goal = have_goal_;
            current_path = current_path_;  // Copy current path
        }

        if (have_pose) {
            live_pts = transformPointsToMap(live_pts, start_x, start_y, start_z, start_yaw);
        } else {
            live_pts.clear();
        }

        double z_min = get_parameter("z_min").as_double();
        double z_max = get_parameter("z_max").as_double();
        cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Live grid before overlay: %d x %d (cols x rows)",
            live_grid.cols, live_grid.rows);
        cv::Mat combined = bridge_.mergeWithStaticMap(live_grid);

        // Publish current combined occupancy grid for visualization (map frame: origin + resolution)
        nav_msgs::msg::OccupancyGrid occ_msg;
        occ_msg.header.frame_id = "map";
        occ_msg.header.stamp = now();
        occ_msg.info.resolution = static_cast<float>(bridge_.resolution());
        occ_msg.info.width = combined.cols;
        occ_msg.info.height = combined.rows;
        occ_msg.info.origin.position.x = bridge_.xMin();
        occ_msg.info.origin.position.y = bridge_.yMin();
        occ_msg.info.origin.position.z = 0.0;
        occ_msg.info.origin.orientation.w = 1.0;
        occ_msg.data.resize(combined.rows * combined.cols);
        for (size_t i = 0; i < occ_msg.data.size(); ++i)
            occ_msg.data[i] = combined.data[i] ? 100 : 0;  // OccupancyGrid: 0=free, 100=occupied
        pub_occ_grid_->publish(occ_msg);

        if (!have_pose || !have_goal) {
            return;
        }

        // Check if current path intersects obstacles within lookahead distance
        bool needs_replan = current_path.empty();
        if (!needs_replan) {
            double lookahead = get_parameter("replan_lookahead_distance").as_double();
            needs_replan = pathIntersectsObstacles(current_path, start_x, start_y, combined, lookahead);
            if (needs_replan) {
                RCLCPP_INFO(get_logger(), "Path intersects obstacles within %.2f m, replanning...", lookahead);
            }
        }
        // Force replan periodically (e.g. when manually driving so path drifts from robot)
        if (!needs_replan) {
            double interval_sec = get_parameter("replan_interval_sec").as_double();
            if (interval_sec > 0.0) {
                double elapsed = (now() - last_plan_time_).seconds();
                if (last_plan_time_.nanoseconds() == 0 || elapsed >= interval_sec) {
                    needs_replan = true;
                    RCLCPP_DEBUG(get_logger(), "Replan clock (%.1f s interval): replanning.", interval_sec);
                }
            }
        }

        // Only plan if we need a new path
        if (!needs_replan) {
            return;  // Current path is still valid
        }

        // RRT expects 0=free, 255=obstacle (same as our combined)
        // Convert meters to pixels using grid resolution
        double resolution = bridge_.resolution();
        double robot_radius_m = get_parameter("robot_radius").as_double();
        int robot_radius_px = static_cast<int>(std::round(robot_radius_m / resolution));
        int n_iter = get_parameter("rrt_iterations").as_int();
        double step_size_m = get_parameter("rrt_step_size").as_double();
        double step_size_px = step_size_m / resolution;  // Convert meters to pixels
        double goal_rate = get_parameter("rrt_goal_sample_rate").as_double();
        int sample_col_min = get_parameter("sample_col_min").as_int();
        int sample_col_max = get_parameter("sample_col_max").as_int();
        int sample_row_min = get_parameter("sample_row_min").as_int();
        int sample_row_max = get_parameter("sample_row_max").as_int();
        bool goal_in_pixels = get_parameter("goal_in_pixels").as_bool();
        std::string planner_type = get_parameter("planner").as_string();

        int start_col, start_row, goal_col, goal_row;
        bridge_.worldToGrid(start_x, start_y, start_col, start_row);
        if (goal_in_pixels) {
            goal_col = static_cast<int>(std::round(goal_x));
            goal_row = static_cast<int>(std::round(goal_y));
        } else {
            bridge_.worldToGrid(goal_x, goal_y, goal_col, goal_row);
        }

        cv::Point2f start_g(static_cast<float>(start_col), static_cast<float>(start_row));
        cv::Point2f goal_g(static_cast<float>(goal_col), static_cast<float>(goal_row));

        if (start_col < 0 || start_col >= bridge_.gridWidth() || start_row < 0 || start_row >= bridge_.gridHeight()) {
            RCLCPP_WARN(get_logger(), "Start out of grid bounds.");
            return;
        }
        if (goal_col < 0 || goal_col >= bridge_.gridWidth() || goal_row < 0 || goal_row >= bridge_.gridHeight()) {
            RCLCPP_WARN(get_logger(), "Goal out of grid bounds.");
            return;
        }

        std::vector<cv::Point2i> path_idx;
        if (planner_type == "astar" || planner_type == "astar_energy") {
            std::vector<double> ps;
            rclcpp::Parameter param = get_parameter("planner_settings");
            if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
                ps = param.as_double_array();
            } else {
                std::string s = param.as_string();
                std::istringstream ss(s);
                std::string token;
                while (std::getline(ss, token, ',') && ps.size() < 4) {
                    try {
                        ps.push_back(std::stod(token));
                    } catch (...) { break; }
                }
            }
            double beta_v = (ps.size() > 0) ? ps[0] : 0.1;
            double smooth_alpha = (ps.size() > 1) ? ps[1] : 0.1;
            double smooth_beta = (ps.size() > 2) ? ps[2] : 0.2;
            int smooth_n_iter = (ps.size() > 3) ? static_cast<int>(ps[3]) : 50;
            path_planning::AStarEnergyPlanner planner(combined, robot_radius_px);
            path_idx = planner.plan(start_g, goal_g, beta_v, smooth_alpha, smooth_beta,
                                    smooth_n_iter, step_size_px);
        } else {
            path_planning::RRTPlanner planner(combined, step_size_px, goal_rate, robot_radius_px, 30.0,
                                             sample_col_min, sample_col_max, sample_row_min, sample_row_max);
            path_idx = planner.plan(start_g, goal_g, n_iter, false, false);
        }

        if (path_idx.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "%s found no path.",
                (planner_type == "astar" || planner_type == "astar_energy") ? "A*" : "RRT");
            return;
        }

        WaypointArray waypoints = std::vector<std::array<double, 2>>();
        for (const auto& pt : path_idx) {
            double x, y;
            bridge_.gridToWorld(pt.x, pt.y, x, y);
            waypoints.push_back({{x, y}});
        }

        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = now();
        for (const auto& wp : waypoints) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose.position.x = wp[0];
            pose.pose.position.y = wp[1];
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
        }
        pub_path_->publish(path_msg);
        pub_waypoints_->publish(path_msg);

        // Store current path for future intersection checks
        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_path_ = waypoints;
        }

        last_plan_time_ = now();

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Published %zu waypoints.", waypoints.size());
    }

    OccGridBridge bridge_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_live_cloud_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_waypoints_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_occ_grid_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    std::mutex mutex_;
    std::vector<std::array<float, 3>> live_points_;
    double start_x_ = 0, start_y_ = 0, goal_x_ = 0, goal_y_ = 0;
    double start_z_ = 0;
    double start_yaw_ = 0;
    bool have_pose_ = false, have_goal_ = false;
    std::vector<std::array<double, 2>> current_path_;  // Current planned path in world coordinates
    rclcpp::Time last_plan_time_{0};  // Time of last successful plan (for replan clock)
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PathPlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

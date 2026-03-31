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
#include <std_msgs/msg/string.hpp>

#include "path_planning/msg/occupancy_grid_array.hpp"
#include "path_planning/msg/agent_centered_input.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <deque>
#include <mutex>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "occ_grid_bridge.hpp"
#include "rrt_planner.hpp"
#include "astar_energy_planner.hpp"

// Placeholder: replace with your waypoint message type if different (e.g. nav_msgs/Path or custom)
using WaypointArray = std::vector<std::array<double, 2>>;

/** One ego-centered input frame: grid + pose + time for motion computation. */
struct EgoFrame {
    cv::Mat grid;
    double x = 0, y = 0, yaw = 0;
    rclcpp::Time time{0};
};

/** Canvas size for map_frame_model: crop is centered on this (1216 x 1216). */
static constexpr int MAP_FRAME_CANVAS_SIZE = 1216;

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
        declare_parameter<double>("robot_radius", 0.05);
        // Radius (m) of circle to clear obstacles from occupancy grid (e.g. wifi adapter sticking up). 0 = disabled.
        declare_parameter<double>("origin_crop_radius", 0.0);
        // Forward offset (m) of the crop circle center from robot origin along +x (robot forward).
        declare_parameter<double>("origin_crop_forward_offset", 0.0);
        // Additional half-width (m) to enforce a corridor around the A* centerline.
        declare_parameter<double>("astar_corridor_half_width", 0.0);
        // Number of RRT* planning iterations (more = better path but slower)
        declare_parameter<int>("rrt_iterations", 10000);
        // RRT* step size in meters (distance to extend tree per iteration, converted to pixels internally)
        declare_parameter<double>("step_size", 0.4);
        // Goal sample rate (probability 0-1): fraction of iterations that sample the goal directly
        declare_parameter<double>("rrt_goal_sample_rate", 0.05);
        // Lookahead distance in meters: if obstacle intersection is beyond this distance along path, don't replan
        declare_parameter<double>("replan_lookahead_distance", 4.0);
        // Force replan every N seconds (e.g. when manually driving so path drifts; 0 = disable periodic replan)
        declare_parameter<double>("replan_interval_sec", 5.0);
        // RRT sampling bounds (grid indices); -1 = use full grid
        declare_parameter<int>("sample_col_min", -1);
        declare_parameter<int>("sample_col_max", -1);
        declare_parameter<int>("sample_row_min", -1);
        declare_parameter<int>("sample_row_max", -1);
        declare_parameter<bool>("goal_in_pixels", false);
        declare_parameter<std::string>("planner", "rrt");
        declare_parameter("planner_settings", std::string("0.1,0.1,0.2,50"));
        declare_parameter<std::string>("planner_status_topic", "/planner_status");

        declare_parameter<std::string>("occ_data_mode", "live");
        declare_parameter<bool>("overlay_live_scans_with_model", false);
        declare_parameter<double>("prediction_temperature", 1.0);
        declare_parameter<int>("num_predicted_frames", 5);
        declare_parameter<double>("model_occupancy_threshold", 0.5);
        declare_parameter<std::string>("model_occ_input_topic", "/map_updater/occ_grid_input");
        declare_parameter<std::string>("model_predicted_output_topic", "/map_updater/predicted_grid_output");
        declare_parameter<int>("agent_frame_stride", 5);
        declare_parameter<int>("plan_interval_ms", 100);
        // Waypoint spacing in meters for resampling planner output (arc-length)
        declare_parameter<double>("waypoint_spacing", 0.4);
        // Local replanning toggle: if true, intersection-triggered replans only replan within local_replan_radius
        declare_parameter<bool>("local_replan_enabled", false);
        // Radius (m) for local replanning: only replan path within this distance from robot
        declare_parameter<double>("local_replan_radius", 5.0);
        // Maximum distance (m) from robot for live scan points; 0 = no limit
        declare_parameter<double>("live_scan_radius", 0.0);

        std::string map_pcd = get_parameter("map_pcd_path").as_string();
        std::string map_png = get_parameter("map_png_path").as_string();
        if (map_pcd.empty() || map_png.empty()) {
            RCLCPP_WARN(get_logger(), "map_pcd_path or map_png_path not set; bridge will not be ready.");
        } else {
            initBridge(map_pcd, map_png);
        }

        rclcpp::SensorDataQoS sensor_qos;
        rclcpp::QoS qos_volatile(1);
        qos_volatile.reliable().durability_volatile();
        rclcpp::QoS qos_be(1);
        qos_be.best_effort().durability_volatile();

        sub_live_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_map", sensor_qos, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                onPointCloud(msg);
            });
        sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/pcl_pose", sensor_qos, [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                onPose(msg);
            });
        sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", qos_be, [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                onGoal(msg);
            });

        std::string mode = get_parameter("occ_data_mode").as_string();
        if (mode == "map_frame_model" || mode == "agent_centered_model") {
            std::string input_topic = get_parameter("model_occ_input_topic").as_string();
            std::string output_topic = get_parameter("model_predicted_output_topic").as_string();
            rclcpp::QoS qos_model(1);
            qos_model.best_effort().durability_volatile();
            pub_model_input_ = create_publisher<path_planning::msg::AgentCenteredInput>(input_topic, qos_model);
            sub_model_output_agent_ = create_subscription<path_planning::msg::AgentCenteredInput>(
                output_topic, qos_model, [this](const path_planning::msg::AgentCenteredInput::SharedPtr msg) {
                    onPredictedAgentCentered(msg);
                });
        }

        pub_path_ = create_publisher<nav_msgs::msg::Path>("/planned_path", qos_be);
        pub_waypoints_ = create_publisher<nav_msgs::msg::Path>("/waypoints", qos_be);
        pub_occ_grid_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_grid", qos_be);
        pub_live_grid_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/live_obstacles", qos_volatile);
        pub_planner_status_ = create_publisher<std_msgs::msg::String>(
            get_parameter("planner_status_topic").as_string(), qos_be);

        bool model_mode = (mode == "agent_centered_model" || mode == "map_frame_model");
        int interval_ms = model_mode ? 100 : get_parameter("plan_interval_ms").as_int();
        RCLCPP_INFO(get_logger(), "Plan interval: %d ms (%s)", interval_ms,
            model_mode ? "model mode, matches 10 Hz lidar" : "plan_interval_ms");
        plan_timer_ = create_wall_timer(
            std::chrono::milliseconds(interval_ms),
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
                "Dimension mismatch: map PCD grid is %d x %d but PNG is %d x %d. Using PNG dimensions.",
                pcd_grid_w, pcd_grid_h, bridge_.gridWidth(), bridge_.gridHeight());
        }
        RCLCPP_INFO(get_logger(), "Bridge initialized: grid %d x %d", bridge_.gridWidth(), bridge_.gridHeight());
    }

    void onPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // auto t_start = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        live_points_.clear();
        live_points_.reserve(static_cast<size_t>(msg->width * msg->height));
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
        new_lidar_data_ = true;
        // auto t_end = std::chrono::high_resolution_clock::now();
        // double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
        //     "[onPointCloud] parsed %d pts in %.1f ms", num, ms);
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
        start_roll_ = roll;
        start_pitch_ = pitch;
        start_yaw_ = yaw;
        have_pose_ = true;
    }

    void onGoal(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        goal_x_ = msg->pose.position.x;
        goal_y_ = msg->pose.position.y;
        have_goal_ = true;
        goal_changed_ = true;
        RCLCPP_INFO(get_logger(), "Received goal pose: x=%.3f, y=%.3f", goal_x_, goal_y_);
    }

    void onPredictedGrids(const path_planning::msg::OccupancyGridArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_predicted_array_ = msg;
    }

    void onPredictedAgentCentered(const path_planning::msg::AgentCenteredInput::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_predicted_agent_ = msg;
    }

    static std::vector<double> boltmannWeights(int n, double T) {
        if (n <= 0 || T <= 0) return {};
        std::vector<double> w(n);
        double Z = 0;
        for (int i = 0; i < n; ++i) {
            w[i] = std::exp(-static_cast<double>(i) / T);
            Z += w[i];
        }
        if (Z > 0)
            for (int i = 0; i < n; ++i) w[i] /= Z;
        return w;
    }

    static cv::Mat weightCombineGrids(const std::vector<cv::Mat>& grids, const std::vector<double>& weights) {
        if (grids.empty() || weights.size() != grids.size()) return cv::Mat();
        const int h = grids[0].rows, w = grids[0].cols;
        cv::Mat sum = cv::Mat::zeros(h, w, CV_64F);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> sum_map(
            sum.ptr<double>(), h, w);
        for (size_t k = 0; k < grids.size(); ++k) {
            if (grids[k].rows != h || grids[k].cols != w) continue;
            Eigen::Map<const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> g_map(
                grids[k].ptr<uchar>(), h, w);
            sum_map += weights[k] * g_map.cast<double>();
        }
        cv::Mat out(h, w, CV_8UC1);
        const double* sum_ptr = sum.ptr<double>();
        uchar* out_ptr = out.ptr<uchar>();
        for (int i = 0; i < h * w; ++i)
            out_ptr[i] = sum_ptr[i] > 0 ? 255 : 0;
        return out;
    }

    static std::vector<cv::Mat> agentCenteredInputToMats(const path_planning::msg::AgentCenteredInput& msg) {
        const int sz = OccGridBridge::EGO_GRID_SIZE;
        const size_t n = static_cast<size_t>(sz * sz);
        std::vector<cv::Mat> out;
        const std::vector<float>* occs[] = {&msg.occ_0, &msg.occ_1, &msg.occ_2, &msg.occ_3, &msg.occ_4};
        for (int i = 0; i < 5; ++i) {
            if (occs[i]->size() != n) continue;
            cv::Mat m(sz, sz, CV_8UC1);
            const float* p = occs[i]->data();
            for (int r = 0; r < sz; ++r) {
                uchar* row = m.ptr<uchar>(r);
                for (int c = 0; c < sz; ++c)
                    row[c] = p[static_cast<size_t>(r * sz + c)] > 0.5f ? 255 : 0;
            }
            out.push_back(m);
        }
        return out;
    }

    static cv::Mat occupancyGridToMat(const nav_msgs::msg::OccupancyGrid& msg, double threshold) {
        int w = static_cast<int>(msg.info.width);
        int h = static_cast<int>(msg.info.height);
        if (w <= 0 || h <= 0 || msg.data.size() != static_cast<size_t>(w * h)) return cv::Mat();
        cv::Mat m(h, w, CV_8UC1);
        for (int i = 0; i < h * w; ++i) {
            int8_t v = msg.data[i];
            if (v >= 0 && v < 100) {
                double threshold_99 = threshold * 99.0;
                if (v >= threshold_99)
                    m.at<uchar>(i / w, i % w) = static_cast<uchar>((v * 255) / 99);
                else
                    m.at<uchar>(i / w, i % w) = 0;
            } else if (v == 100) {
                m.at<uchar>(i / w, i % w) = 255;
            } else {
                m.at<uchar>(i / w, i % w) = 0;
            }
        }
        return m;
    }

    void publishOccupancyGrid(const cv::Mat& grid, const std::string& frame_id = "map",
                             double origin_x = 0, double origin_y = 0, double resolution = 0.05) {
        if (grid.empty()) return;
        nav_msgs::msg::OccupancyGrid occ_msg;
        occ_msg.header.frame_id = frame_id;
        occ_msg.header.stamp = now();
        occ_msg.info.resolution = static_cast<float>(resolution);
        occ_msg.info.width = grid.cols;
        occ_msg.info.height = grid.rows;
        occ_msg.info.origin.position.x = origin_x;
        occ_msg.info.origin.position.y = origin_y;
        occ_msg.info.origin.position.z = 0.0;
        occ_msg.info.origin.orientation.w = 1.0;
        occ_msg.data.resize(static_cast<size_t>(grid.rows * grid.cols));
        bool have_pred = (!pred_only_.empty() && pred_only_.rows == grid.rows && pred_only_.cols == grid.cols);
        for (int r = 0; r < grid.rows; ++r)
            for (int c = 0; c < grid.cols; ++c) {
                size_t idx = static_cast<size_t>(r * grid.cols + c);
                if (grid.at<uchar>(r, c) < 50) {
                    occ_msg.data[idx] = 0;
                } else if (have_pred && pred_only_.at<uchar>(r, c) >= 50) {
                    occ_msg.data[idx] = 101;
                } else {
                    occ_msg.data[idx] = 100;
                }
            }
        pub_occ_grid_->publish(occ_msg);
    }

    void publishLiveGrid(const cv::Mat& grid, const std::string& frame_id = "map",
                         double origin_x = 0, double origin_y = 0, double resolution = 0.05) {
        if (grid.empty() || !pub_live_grid_) return;
        nav_msgs::msg::OccupancyGrid occ_msg;
        occ_msg.header.frame_id = frame_id;
        occ_msg.header.stamp = now();
        occ_msg.info.resolution = static_cast<float>(resolution);
        occ_msg.info.width = grid.cols;
        occ_msg.info.height = grid.rows;
        occ_msg.info.origin.position.x = origin_x;
        occ_msg.info.origin.position.y = origin_y;
        occ_msg.info.origin.position.z = 0.0;
        occ_msg.info.origin.orientation.w = 1.0;
        occ_msg.data.resize(static_cast<size_t>(grid.rows * grid.cols));
        for (int r = 0; r < grid.rows; ++r)
            for (int c = 0; c < grid.cols; ++c)
                occ_msg.data[static_cast<size_t>(r * grid.cols + c)] = (grid.at<uchar>(r, c) >= 50) ? 100 : 0;
        pub_live_grid_->publish(occ_msg);
    }

    void publishPlannerStatus(const std::string& planner_type,
                              const std::string& status,
                              const std::string& reason,
                              size_t path_points,
                              bool intersection,
                              double planning_ms,
                              double cycle_ms) {
        if (!pub_planner_status_) return;
        std_msgs::msg::String msg;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1);
        ss << "Planner: " << planner_type << "\n";
        ss << "Status: " << status << "\n";
        ss << "Reason: " << reason << "\n";
        ss << "Path points: " << path_points << "\n";
        ss << "Intersection: " << (intersection ? "yes" : "no") << "\n";
        if (planning_ms >= 0.0) ss << "Planning: " << planning_ms << " ms\n";
        else ss << "Planning: n/a\n";
        if (cycle_ms >= 0.0) ss << "Cycle: " << cycle_ms << " ms";
        else ss << "Cycle: n/a";
        msg.data = ss.str();
        pub_planner_status_->publish(msg);
    }

    cv::Mat cropMapAndCenterOnCanvas(const cv::Mat& full_grid,
                                     int sample_col_min, int sample_col_max,
                                     int sample_row_min, int sample_row_max) const {
        const int w = bridge_.gridWidth();
        const int h = bridge_.gridHeight();
        if (full_grid.empty() || full_grid.cols != w || full_grid.rows != h) return cv::Mat();
        int sc_min = (sample_col_min >= 0) ? std::max(0, sample_col_min) : 0;
        int sc_max = (sample_col_max >= 0) ? std::min(w - 1, sample_col_max) : w - 1;
        int sr_min = (sample_row_min >= 0) ? std::max(0, sample_row_min) : 0;
        int sr_max = (sample_row_max >= 0) ? std::min(h - 1, sample_row_max) : h - 1;
        if (sc_min > sc_max || sr_min > sr_max) return cv::Mat();
        const int crop_w = std::min(sc_max - sc_min + 1, MAP_FRAME_CANVAS_SIZE);
        const int crop_h = std::min(sr_max - sr_min + 1, MAP_FRAME_CANVAS_SIZE);
        cv::Mat crop = full_grid(cv::Rect(sc_min, sr_min, crop_w, crop_h)).clone();
        cv::Mat canvas = cv::Mat::zeros(MAP_FRAME_CANVAS_SIZE, MAP_FRAME_CANVAS_SIZE, CV_8UC1);
        const int cx = (MAP_FRAME_CANVAS_SIZE - crop_w) / 2;
        const int cy = (MAP_FRAME_CANVAS_SIZE - crop_h) / 2;
        crop.copyTo(canvas(cv::Rect(cx, cy, crop_w, crop_h)));
        return canvas;
    }

    void cookieCutterCanvasCropOntoMap(const cv::Mat& canvas_1216,
                                      int sample_col_min, int sample_col_max,
                                      int sample_row_min, int sample_row_max,
                                      cv::Mat& map_out) const {
        const int w = bridge_.gridWidth();
        const int h = bridge_.gridHeight();
        if (canvas_1216.empty() || canvas_1216.cols != MAP_FRAME_CANVAS_SIZE ||
            canvas_1216.rows != MAP_FRAME_CANVAS_SIZE || map_out.empty() ||
            map_out.cols != w || map_out.rows != h)
            return;
        int sc_min = (sample_col_min >= 0) ? std::max(0, sample_col_min) : 0;
        int sc_max = (sample_col_max >= 0) ? std::min(w - 1, sample_col_max) : w - 1;
        int sr_min = (sample_row_min >= 0) ? std::max(0, sample_row_min) : 0;
        int sr_max = (sample_row_max >= 0) ? std::min(h - 1, sample_row_max) : h - 1;
        if (sc_min > sc_max || sr_min > sr_max) return;
        const int crop_w = std::min(sc_max - sc_min + 1, MAP_FRAME_CANVAS_SIZE);
        const int crop_h = std::min(sr_max - sr_min + 1, MAP_FRAME_CANVAS_SIZE);
        const int cx = (MAP_FRAME_CANVAS_SIZE - crop_w) / 2;
        const int cy = (MAP_FRAME_CANVAS_SIZE - crop_h) / 2;
        for (int r = sr_min; r <= sr_max; ++r)
            for (int c = sc_min; c <= sc_max; ++c)
                map_out.at<uchar>(r, c) = 0;
        for (int r = 0; r < crop_h; ++r)
            for (int c = 0; c < crop_w; ++c) {
                int mr = sr_min + r;
                int mc = sc_min + c;
                if (mr >= 0 && mr < h && mc >= 0 && mc < w)
                    map_out.at<uchar>(mr, mc) = canvas_1216.at<uchar>(cy + r, cx + c);
            }
    }

    path_planning::msg::AgentCenteredInput toAgentCenteredInputMapFrame(const std::deque<cv::Mat>& queue) const {
        path_planning::msg::AgentCenteredInput msg;
        msg.header.frame_id = "map";
        msg.header.stamp = now();
        msg.mode = 0;
        const int sz = MAP_FRAME_CANVAS_SIZE;
        const size_t n = static_cast<size_t>(sz * sz);
        if (queue.size() != 5u) return msg;
        for (int i = 0; i < 5; ++i) {
            const cv::Mat& g = queue[i];
            if (g.empty() || g.cols != sz || g.rows != sz) continue;
            std::vector<float>* occ[] = {&msg.occ_0, &msg.occ_1, &msg.occ_2, &msg.occ_3, &msg.occ_4};
            occ[i]->resize(n);
            float* dst = occ[i]->data();
            for (int r = 0; r < sz; ++r) {
                const uchar* row = g.ptr<uchar>(r);
                for (int c = 0; c < sz; ++c)
                    dst[static_cast<size_t>(r * sz + c)] = row[c] ? 1.f : 0.f;
            }
        }
        return msg;
    }

    path_planning::msg::AgentCenteredInput toAgentCenteredInput(const std::deque<EgoFrame>& queue) const {
        path_planning::msg::AgentCenteredInput msg;
        msg.header.frame_id = "map";
        msg.header.stamp = now();
        msg.mode = 1;
        const int sz = OccGridBridge::EGO_GRID_SIZE;
        const size_t n = static_cast<size_t>(sz * sz);

        if (queue.size() != 5u) return msg;

        msg.occ_0.resize(n); msg.occ_1.resize(n); msg.occ_2.resize(n); msg.occ_3.resize(n); msg.occ_4.resize(n);
        msg.delta_0.assign(n, 0.f);
        msg.delta_1.resize(n); msg.delta_2.resize(n); msg.delta_3.resize(n); msg.delta_4.resize(n);

        for (int r = 0; r < sz; ++r) {
            const uchar* p0 = queue[0].grid.ptr<uchar>(r);
            const uchar* p1 = queue[1].grid.ptr<uchar>(r);
            const uchar* p2 = queue[2].grid.ptr<uchar>(r);
            const uchar* p3 = queue[3].grid.ptr<uchar>(r);
            const uchar* p4 = queue[4].grid.ptr<uchar>(r);
            float* o0 = msg.occ_0.data() + static_cast<size_t>(r) * sz;
            float* o1 = msg.occ_1.data() + static_cast<size_t>(r) * sz;
            float* o2 = msg.occ_2.data() + static_cast<size_t>(r) * sz;
            float* o3 = msg.occ_3.data() + static_cast<size_t>(r) * sz;
            float* o4 = msg.occ_4.data() + static_cast<size_t>(r) * sz;
            float* d1 = msg.delta_1.data() + static_cast<size_t>(r) * sz;
            float* d2 = msg.delta_2.data() + static_cast<size_t>(r) * sz;
            float* d3 = msg.delta_3.data() + static_cast<size_t>(r) * sz;
            float* d4 = msg.delta_4.data() + static_cast<size_t>(r) * sz;
            for (int c = 0; c < sz; ++c) {
                float v0 = p0[c] ? 1.f : 0.f, v1 = p1[c] ? 1.f : 0.f, v2 = p2[c] ? 1.f : 0.f;
                float v3 = p3[c] ? 1.f : 0.f, v4 = p4[c] ? 1.f : 0.f;
                o0[c] = v0; o1[c] = v1; o2[c] = v2; o3[c] = v3; o4[c] = v4;
                d1[c] = v1 - v0; d2[c] = v2 - v1; d3[c] = v3 - v2; d4[c] = v4 - v3;
            }
        }

        msg.motion_forward_speed.resize(5);
        msg.motion_yaw_rate.resize(5);
        msg.motion_forward_speed[0] = 0.f;
        msg.motion_yaw_rate[0] = 0.f;
        for (size_t j = 1; j < 5u; ++j) {
            const EgoFrame& curr = queue[j];
            const EgoFrame& prev = queue[j - 1];
            double dt = (curr.time - prev.time).seconds();
            if (dt < 1e-6) dt = 1e-6;
            double dx = curr.x - prev.x;
            double dy = curr.y - prev.y;
            double cy = std::cos(curr.yaw);
            double sy = std::sin(curr.yaw);
            double body_x = cy * dx + sy * dy;
            double dyaw = curr.yaw - prev.yaw;
            while (dyaw > 3.141592653589793) dyaw -= 2.0 * 3.141592653589793;
            while (dyaw < -3.141592653589793) dyaw += 2.0 * 3.141592653589793;
            msg.motion_forward_speed[j] = static_cast<float>(body_x / dt);
            msg.motion_yaw_rate[j] = static_cast<float>(dyaw / dt);
        }
        return msg;
    }

    WaypointArray resampleWaypointsWorld(const WaypointArray& path, double spacing) const {
        if (path.size() < 2 || spacing <= 0) return path;
        std::vector<double> arc(path.size(), 0.0);
        for (size_t i = 1; i < path.size(); ++i) {
            double dx = path[i][0] - path[i - 1][0];
            double dy = path[i][1] - path[i - 1][1];
            arc[i] = arc[i - 1] + std::sqrt(dx * dx + dy * dy);
        }
        double total = arc.back();
        if (total < 1e-9) return path;
        WaypointArray out;
        out.push_back(path.front());
        size_t seg = 0;
        for (double s = spacing; s < total; s += spacing) {
            while (seg + 1 < arc.size() - 1 && arc[seg + 1] < s) ++seg;
            double seg_len = std::max(arc[seg + 1] - arc[seg], 1e-9);
            double t = (s - arc[seg]) / seg_len;
            double x = path[seg][0] + t * (path[seg + 1][0] - path[seg][0]);
            double y = path[seg][1] + t * (path[seg + 1][1] - path[seg][1]);
            out.push_back({{x, y}});
        }
        out.push_back(path.back());
        return out;
    }

    cv::Mat computeDistanceTransform(const cv::Mat& occupancy_grid) const {
        cv::Mat binary_inv(occupancy_grid.rows, occupancy_grid.cols, CV_8UC1);
        for (int r = 0; r < occupancy_grid.rows; ++r) {
            const uchar* src = occupancy_grid.ptr<uchar>(r);
            uchar* dst = binary_inv.ptr<uchar>(r);
            for (int c = 0; c < occupancy_grid.cols; ++c) {
                dst[c] = (src[c] < 50) ? 255 : 0;
            }
        }
        cv::Mat dist;
        cv::distanceTransform(binary_inv, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        dist.convertTo(dist, CV_64F);
        return dist;
    }

    bool pathIntersectsObstacles(const std::vector<std::array<double, 2>>& path_world,
                                 double robot_x, double robot_y,
                                 const cv::Mat& dist,
                                 double lookahead_distance,
                                 double required_clearance_px) const {
        if (path_world.empty()) return false;
        const double resolution = bridge_.resolution();

        double cumulative_dist = 0.0;
        const Eigen::Vector2d robot(robot_x, robot_y);

        size_t start_idx = 0;
        double min_dist_sq = 1e30;
        for (size_t i = 0; i < path_world.size(); ++i) {
            Eigen::Vector2d p(path_world[i][0], path_world[i][1]);
            double d_sq = (p - robot).squaredNorm();
            if (d_sq < min_dist_sq) {
                min_dist_sq = d_sq;
                start_idx = i;
            }
        }

        for (size_t i = start_idx; i + 1 < path_world.size(); ++i) {
            Eigen::Vector2d p0(path_world[i][0], path_world[i][1]);
            Eigen::Vector2d p1(path_world[i + 1][0], path_world[i + 1][1]);

            double seg_len = (p1 - p0).norm();
            if (i > start_idx) {
                cumulative_dist += seg_len;
            } else {
                cumulative_dist = (p0 - robot).norm();
            }
            if (cumulative_dist > lookahead_distance) break;

            int col0, row0, col1, row1;
            bridge_.worldToGrid(p0.x(), p0.y(), col0, row0);
            bridge_.worldToGrid(p1.x(), p1.y(), col1, row1);

            int num_samples = static_cast<int>(seg_len / resolution) + 1;
            for (int j = 0; j <= num_samples; ++j) {
                double t = static_cast<double>(j) / num_samples;
                int col = static_cast<int>(col0 + t * (col1 - col0));
                int row = static_cast<int>(row0 + t * (row1 - row0));
                if (col >= 0 && col < dist.cols && row >= 0 && row < dist.rows) {
                    if (dist.at<double>(row, col) <= required_clearance_px) return true;
                }
            }
        }
        return false;
    }

    std::vector<std::array<float, 3>> transformPointsToMap(
        const std::vector<std::array<float, 3>>& points_lidar,
        double tx, double ty, double tz,
        double roll, double pitch, double yaw) const {
        if (points_lidar.empty()) return {};
        const size_t n = points_lidar.size();

        Eigen::AngleAxisd roll_angle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R = (yaw_angle * pitch_angle * roll_angle).toRotationMatrix();
        Eigen::Vector3d t(tx, ty, tz);

        Eigen::Matrix3Xd P(3, n);
        for (size_t i = 0; i < n; ++i) {
            P(0, i) = static_cast<double>(points_lidar[i][0]);
            P(1, i) = static_cast<double>(points_lidar[i][1]);
            P(2, i) = static_cast<double>(points_lidar[i][2]);
        }
        Eigen::Matrix3Xd P_map = (R * P).colwise() + t;

        std::vector<std::array<float, 3>> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i)
            out.push_back({{static_cast<float>(P_map(0, i)), static_cast<float>(P_map(1, i)), static_cast<float>(P_map(2, i))}});
        return out;
    }

    void runPlanningCycle() {
        if (!bridge_.isInitialized()) return;

        auto t_cycle_start = std::chrono::high_resolution_clock::now();

        std::vector<std::array<float, 3>> live_pts;
        bool has_new_lidar_data = false;
        bool goal_changed = false;
        double start_x, start_y, goal_x, goal_y;
        double start_z, start_roll, start_pitch, start_yaw;
        bool have_pose, have_goal;
        std::vector<std::array<double, 2>> current_path;
        path_planning::msg::OccupancyGridArray::SharedPtr predicted_msg;
        path_planning::msg::AgentCenteredInput::SharedPtr predicted_agent_msg;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            has_new_lidar_data = new_lidar_data_;
            if (has_new_lidar_data) {
                live_points_staging_.swap(live_points_);
                new_lidar_data_ = false;
            }
            goal_changed = goal_changed_;
            goal_changed_ = false;
            start_x = start_x_;
            start_y = start_y_;
            start_z = start_z_;
            start_roll = start_roll_;
            start_pitch = start_pitch_;
            start_yaw = start_yaw_;
            goal_x = goal_x_;
            goal_y = goal_y_;
            have_pose = have_pose_;
            have_goal = have_goal_;
            current_path = current_path_;
            predicted_msg = last_predicted_array_;
            predicted_agent_msg = last_predicted_agent_;
        }
        if (has_new_lidar_data) {
            live_pts = std::move(live_points_staging_);
        } else {
            live_pts.clear();
        }

        auto t_after_swap = std::chrono::high_resolution_clock::now();

        std::string mode = get_parameter("occ_data_mode").as_string();
        bool overlay_live_scans_with_model = get_parameter("overlay_live_scans_with_model").as_bool();
        double z_min = get_parameter("z_min").as_double();
        double z_max = get_parameter("z_max").as_double();

        if (has_new_lidar_data) {
            if (have_pose) {
                live_pts = transformPointsToMap(live_pts, start_x, start_y, start_z,
                                                start_roll, start_pitch, start_yaw);
                double live_scan_radius = get_parameter("live_scan_radius").as_double();
                if (live_scan_radius > 0.0) {
                    double r2 = live_scan_radius * live_scan_radius;
                    float sx = static_cast<float>(start_x);
                    float sy = static_cast<float>(start_y);
                    auto it = std::remove_if(live_pts.begin(), live_pts.end(),
                        [sx, sy, r2](const std::array<float, 3>& p) {
                            float dx = p[0] - sx, dy = p[1] - sy;
                            return (dx * dx + dy * dy) > static_cast<float>(r2);
                        });
                    live_pts.erase(it, live_pts.end());
                }
            } else {
                live_pts.clear();
            }
        }

        auto t_after_transform = std::chrono::high_resolution_clock::now();

        cv::Mat combined;
        if (!has_new_lidar_data && have_cached_combined_grid_) {
            combined = cached_combined_grid_.clone();
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "No new lidar scan; reusing previous combined occupancy grid.");
        } else {
            if (mode == "live") {
                cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
                combined = bridge_.mergeWithStaticMap(live_grid);
                publishLiveGrid(live_grid, "map", bridge_.xMin(), bridge_.yMin(), bridge_.resolution());
            } else if (mode == "map_frame_model") {
                cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
                if (have_pose) {
                    cv::Mat mf_grid = OccGridBridge::pointcloudToMapFrameOccupancyGrid201(
                        live_pts, start_x, start_y, z_min, z_max);
                    EgoFrame frame;
                    frame.grid = mf_grid;
                    frame.x = start_x;
                    frame.y = start_y;
                    frame.yaw = start_yaw;
                    frame.time = now();
                    const int stride = get_parameter("agent_frame_stride").as_int();
                    const size_t required_queue = static_cast<size_t>(4 * std::max(1, stride) + 1);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        queue_ego_frames_.push_back(frame);
                        while (queue_ego_frames_.size() > required_queue) queue_ego_frames_.pop_front();
                        anchor_x_ = queue_ego_frames_.back().x;
                        anchor_y_ = queue_ego_frames_.back().y;
                        anchor_yaw_ = queue_ego_frames_.back().yaw;
                    }
                    if (pub_model_input_ && queue_ego_frames_.size() == required_queue) {
                        std::deque<EgoFrame> q_full;
                        { std::lock_guard<std::mutex> lock(mutex_); q_full = queue_ego_frames_; }
                        std::deque<EgoFrame> q_strided;
                        for (int s = 0; s < 5; ++s)
                            q_strided.push_back(q_full[static_cast<size_t>(s * std::max(1, stride))]);
                        path_planning::msg::AgentCenteredInput msg = toAgentCenteredInput(q_strided);
                        msg.mode = 0;
                        pub_model_input_->publish(msg);
                    }
                }
                combined = bridge_.mergeWithStaticMap(cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1));
                pred_only_ = cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1);
                if (predicted_agent_msg) {
                    std::vector<cv::Mat> grids = agentCenteredInputToMats(*predicted_agent_msg);
                    if (!grids.empty()) {
                        int N = get_parameter("num_predicted_frames").as_int();
                        double T = get_parameter("prediction_temperature").as_double();
                        size_t n = std::min(static_cast<size_t>(N), grids.size());
                        std::vector<double> w = boltmannWeights(static_cast<int>(n), T);
                        cv::Mat weighted_mf = weightCombineGrids(grids, w);
                        if (!weighted_mf.empty()) {
                            double ax, ay;
                            { std::lock_guard<std::mutex> lock(mutex_); ax = anchor_x_; ay = anchor_y_; }
                            bridge_.pasteMapFrameGridIntoMap(weighted_mf, ax, ay, combined);
                            bridge_.pasteMapFrameGridIntoMap(weighted_mf, ax, ay, pred_only_);
                        }
                    }
                }
                if (overlay_live_scans_with_model && !live_grid.empty()) {
                    combined = bridge_.mergeWithStaticMap(cv::max(combined, live_grid));
                }
                publishLiveGrid(live_grid, "map", bridge_.xMin(), bridge_.yMin(), bridge_.resolution());
            } else if (mode == "agent_centered_model") {
                cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
                if (have_pose) {
                    auto t_before_ego = std::chrono::high_resolution_clock::now();
                    cv::Mat ego_grid = OccGridBridge::pointcloudToEgoOccupancyGrid201(
                        live_pts, start_x, start_y, start_yaw, z_min, z_max);
                    auto t_after_ego = std::chrono::high_resolution_clock::now();
                    EgoFrame frame;
                    frame.grid = ego_grid;
                    frame.x = start_x;
                    frame.y = start_y;
                    frame.yaw = start_yaw;
                    frame.time = now();
                    const int stride = get_parameter("agent_frame_stride").as_int();
                    const size_t required_queue = static_cast<size_t>(4 * std::max(1, stride) + 1);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        queue_ego_frames_.push_back(frame);
                        while (queue_ego_frames_.size() > required_queue) queue_ego_frames_.pop_front();
                        anchor_x_ = queue_ego_frames_.back().x;
                        anchor_y_ = queue_ego_frames_.back().y;
                        anchor_yaw_ = queue_ego_frames_.back().yaw;
                    }
                    if (pub_model_input_ && queue_ego_frames_.size() == required_queue) {
                        std::deque<EgoFrame> q_full;
                        { std::lock_guard<std::mutex> lock(mutex_); q_full = queue_ego_frames_; }
                        std::deque<EgoFrame> q_strided;
                        for (int s = 0; s < 5; ++s)
                            q_strided.push_back(q_full[static_cast<size_t>(s * std::max(1, stride))]);
                        auto t_before_to_msg = std::chrono::high_resolution_clock::now();
                        path_planning::msg::AgentCenteredInput msg = toAgentCenteredInput(q_strided);
                        auto t_after_to_msg = std::chrono::high_resolution_clock::now();
                        pub_model_input_->publish(msg);
                        auto t_after_publish = std::chrono::high_resolution_clock::now();
                        using Ms = std::chrono::duration<double, std::milli>;
                        double ms_swap = std::chrono::duration_cast<Ms>(t_after_swap - t_cycle_start).count();
                        double ms_tf = std::chrono::duration_cast<Ms>(t_after_transform - t_after_swap).count();
                        double ms_ego = std::chrono::duration_cast<Ms>(t_after_ego - t_before_ego).count();
                        double ms_to_msg = std::chrono::duration_cast<Ms>(t_after_to_msg - t_before_to_msg).count();
                        double ms_pub = std::chrono::duration_cast<Ms>(t_after_publish - t_after_to_msg).count();
                        double ms_total = std::chrono::duration_cast<Ms>(t_after_publish - t_cycle_start).count();
                        RCLCPP_INFO(get_logger(),
                            "[agent_centered] swap=%.1fms transform=%.1fms ego_grid=%.1fms to_msg=%.1fms publish=%.1fms | total=%.1fms",
                            ms_swap, ms_tf, ms_ego, ms_to_msg, ms_pub, ms_total);
                    }
                }
                combined = bridge_.mergeWithStaticMap(cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1));
                pred_only_ = cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1);
                if (predicted_agent_msg) {
                    std::vector<cv::Mat> grids = agentCenteredInputToMats(*predicted_agent_msg);
                    if (!grids.empty()) {
                        int N = get_parameter("num_predicted_frames").as_int();
                        double T = get_parameter("prediction_temperature").as_double();
                        size_t n = std::min(static_cast<size_t>(N), grids.size());
                        std::vector<double> w = boltmannWeights(static_cast<int>(n), T);
                        cv::Mat weighted_ego = weightCombineGrids(grids, w);
                        if (!weighted_ego.empty()) {
                            double ax, ay, ayaw;
                            { std::lock_guard<std::mutex> lock(mutex_); ax = anchor_x_; ay = anchor_y_; ayaw = anchor_yaw_; }
                            bridge_.pasteEgoGridIntoMap(weighted_ego, ax, ay, ayaw, combined);
                            bridge_.pasteEgoGridIntoMap(weighted_ego, ax, ay, ayaw, pred_only_);
                        }
                    }
                }
                if (overlay_live_scans_with_model && !live_grid.empty()) {
                    combined = bridge_.mergeWithStaticMap(cv::max(combined, live_grid));
                }
                publishLiveGrid(live_grid, "map", bridge_.xMin(), bridge_.yMin(), bridge_.resolution());
            } else {
                cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
                combined = bridge_.mergeWithStaticMap(live_grid);
            }

            if (!combined.empty()) {
                cached_combined_grid_ = combined.clone();
                have_cached_combined_grid_ = true;
            }
        }

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Combined grid: %d x %d (mode=%s)", combined.cols, combined.rows, mode.c_str());

        // Clear obstacles within origin_crop_radius of a point offset forward from the robot origin
        double crop_radius_m = get_parameter("origin_crop_radius").as_double();
        if (have_pose && crop_radius_m > 0.0 && !combined.empty()) {
            double fwd_offset = get_parameter("origin_crop_forward_offset").as_double();
            double crop_cx = start_x + fwd_offset * std::cos(start_yaw);
            double crop_cy = start_y + fwd_offset * std::sin(start_yaw);
            double res = bridge_.resolution();
            int crop_radius_px = static_cast<int>(std::ceil(crop_radius_m / res));
            int center_col, center_row;
            bridge_.worldToGrid(crop_cx, crop_cy, center_col, center_row);
            int r_min = std::max(0, center_row - crop_radius_px);
            int r_max = std::min(combined.rows - 1, center_row + crop_radius_px);
            int c_min = std::max(0, center_col - crop_radius_px);
            int c_max = std::min(combined.cols - 1, center_col + crop_radius_px);
            double r2 = static_cast<double>(crop_radius_px) * crop_radius_px;
            for (int r = r_min; r <= r_max; ++r) {
                for (int c = c_min; c <= c_max; ++c) {
                    double dr = r - center_row;
                    double dc = c - center_col;
                    if (dr * dr + dc * dc <= r2)
                        combined.at<uchar>(r, c) = 0;
                }
            }
        }

        publishOccupancyGrid(combined, "map", bridge_.xMin(), bridge_.yMin(), bridge_.resolution());

        std::string planner_type = get_parameter("planner").as_string();
        auto cycle_ms_now = [&]() {
            return std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_cycle_start).count();
        };

        if (!have_pose || !have_goal) {
            std::string reason = !have_pose ? "waiting_for_pose" : "waiting_for_goal";
            publishPlannerStatus(planner_type, "idle", reason, current_path.size(), false, -1.0, cycle_ms_now());
            return;
        }

        double resolution = bridge_.resolution();
        double robot_radius_m = get_parameter("robot_radius").as_double();
        int robot_radius_px = static_cast<int>(std::round(std::max(0.0, robot_radius_m) / resolution));
        double astar_corridor_half_width_m = get_parameter("astar_corridor_half_width").as_double();
        int astar_corridor_half_width_px = static_cast<int>(std::round(
            std::max(0.0, astar_corridor_half_width_m) / resolution));
        double required_clearance_px = static_cast<double>(robot_radius_px + astar_corridor_half_width_px);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "planner=%s robot_radius=%.3f m (%d px), astar_half_width=%.3f m (%d px), required_clearance=%.1f px",
            planner_type.c_str(), robot_radius_m, robot_radius_px,
            astar_corridor_half_width_m, astar_corridor_half_width_px, required_clearance_px);

        bool needs_replan = current_path.empty();
        bool intersection_detected = false;
        bool is_local_replan = false;
        std::string replan_reason = current_path.empty() ? "no_current_path" : "tracking";

        // Immediate replan: new goal received
        if (!needs_replan && goal_changed) {
            needs_replan = true;
            replan_reason = "new_goal";
            RCLCPP_INFO(get_logger(), "New goal received, replanning immediately.");
        }
        // Immediate replan: intersection within lookahead
        // Only compute the expensive distance transform when we actually need intersection checking
        cv::Mat dist_map;
        if (!needs_replan && !current_path.empty() && !combined.empty()) {
            dist_map = computeDistanceTransform(combined);
        }
        if (!needs_replan && !dist_map.empty()) {
            double lookahead = get_parameter("replan_lookahead_distance").as_double();
            needs_replan = pathIntersectsObstacles(
                current_path, start_x, start_y, dist_map, lookahead, required_clearance_px);
            if (needs_replan) {
                intersection_detected = true;
                replan_reason = "intersection";
                bool local_enabled = get_parameter("local_replan_enabled").as_bool();
                if (local_enabled) is_local_replan = true;
                RCLCPP_INFO(get_logger(), "Path intersects obstacles within %.2f m, replanning %s...",
                    lookahead, is_local_replan ? "locally" : "globally");
            }
        }
        // Periodic replan (lower priority)
        if (!needs_replan) {
            double interval_sec = get_parameter("replan_interval_sec").as_double();
            if (interval_sec > 0.0) {
                double elapsed = (now() - last_plan_time_).seconds();
                if (last_plan_time_.nanoseconds() == 0 || elapsed >= interval_sec) {
                    needs_replan = true;
                    replan_reason = (last_plan_time_.nanoseconds() == 0) ? "initial_plan" : "periodic_replan";
                }
            }
        }

        if (!needs_replan) {
            publishPlannerStatus(planner_type, "tracking", "path_clear", current_path.size(),
                                 false, -1.0, cycle_ms_now());
            return;
        }

        int n_iter = get_parameter("rrt_iterations").as_int();
        double step_size_m = get_parameter("step_size").as_double();
        double step_size_px = step_size_m / resolution;
        double goal_rate = get_parameter("rrt_goal_sample_rate").as_double();
        int sample_col_min = get_parameter("sample_col_min").as_int();
        int sample_col_max = get_parameter("sample_col_max").as_int();
        int sample_row_min = get_parameter("sample_row_min").as_int();
        int sample_row_max = get_parameter("sample_row_max").as_int();
        bool goal_in_pixels = get_parameter("goal_in_pixels").as_bool();

        // Determine planning goal: for local replan, use a point along the existing path
        double local_replan_radius = get_parameter("local_replan_radius").as_double();
        double local_goal_x = goal_x, local_goal_y = goal_y;
        size_t splice_idx = 0;
        if (is_local_replan && !current_path.empty()) {
            const Eigen::Vector2d robot(start_x, start_y);
            size_t closest_idx = 0;
            double min_d2 = 1e30;
            for (size_t i = 0; i < current_path.size(); ++i) {
                Eigen::Vector2d p(current_path[i][0], current_path[i][1]);
                double d2 = (p - robot).squaredNorm();
                if (d2 < min_d2) { min_d2 = d2; closest_idx = i; }
            }
            double cum = 0.0;
            splice_idx = current_path.size() - 1;
            for (size_t i = closest_idx; i + 1 < current_path.size(); ++i) {
                Eigen::Vector2d p0(current_path[i][0], current_path[i][1]);
                Eigen::Vector2d p1(current_path[i + 1][0], current_path[i + 1][1]);
                cum += (p1 - p0).norm();
                if (cum >= local_replan_radius) {
                    splice_idx = i + 1;
                    break;
                }
            }
            local_goal_x = current_path[splice_idx][0];
            local_goal_y = current_path[splice_idx][1];
            RCLCPP_INFO(get_logger(), "Local replan: goal at path[%zu] (%.3f, %.3f), radius=%.2f m",
                splice_idx, local_goal_x, local_goal_y, local_replan_radius);
        }

        int start_col, start_row, goal_col, goal_row;
        bridge_.worldToGrid(start_x, start_y, start_col, start_row);
        if (!is_local_replan && goal_in_pixels) {
            goal_col = static_cast<int>(std::round(local_goal_x));
            goal_row = static_cast<int>(std::round(local_goal_y));
        } else {
            bridge_.worldToGrid(local_goal_x, local_goal_y, goal_col, goal_row);
        }

        cv::Point2f start_g(static_cast<float>(start_col), static_cast<float>(start_row));
        cv::Point2f goal_g(static_cast<float>(goal_col), static_cast<float>(goal_row));

        if (start_col < 0 || start_col >= bridge_.gridWidth() || start_row < 0 || start_row >= bridge_.gridHeight()) {
            RCLCPP_WARN(get_logger(), "Start out of grid bounds.");
            publishPlannerStatus(planner_type, "error", "start_out_of_bounds", current_path.size(),
                                 intersection_detected, -1.0, cycle_ms_now());
            return;
        }
        if (goal_col < 0 || goal_col >= bridge_.gridWidth() || goal_row < 0 || goal_row >= bridge_.gridHeight()) {
            RCLCPP_WARN(get_logger(), "Goal out of grid bounds.");
            publishPlannerStatus(planner_type, "error", "goal_out_of_bounds", current_path.size(),
                                 intersection_detected, -1.0, cycle_ms_now());
            return;
        }

        auto t_plan_start = std::chrono::high_resolution_clock::now();
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
                    try { ps.push_back(std::stod(token)); } catch (...) { break; }
                }
            }
            double beta_v = (ps.size() > 0) ? ps[0] : 0.1;
            double smooth_alpha = (ps.size() > 1) ? ps[1] : 0.1;
            double smooth_beta = (ps.size() > 2) ? ps[2] : 0.2;
            int smooth_n_iter = (ps.size() > 3) ? static_cast<int>(ps[3]) : 50;
            path_planning::AStarEnergyPlanner planner(
                combined, robot_radius_px, astar_corridor_half_width_px);
            path_idx = planner.plan(start_g, goal_g, beta_v, smooth_alpha, smooth_beta,
                                    smooth_n_iter, step_size_px);
        } else {
            path_planning::RRTPlanner planner(combined, step_size_px, goal_rate, robot_radius_px, 30.0,
                                             sample_col_min, sample_col_max, sample_row_min, sample_row_max);
            path_idx = planner.plan(start_g, goal_g, n_iter, false, false);
        }
        double planning_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_plan_start).count();

        if (path_idx.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s found no path. Publishing stop-in-place.",
                (planner_type == "astar" || planner_type == "astar_energy") ? "A*" : "RRT");
            // Publish single waypoint at robot position so controller stops in place instead of following stale path.
            nav_msgs::msg::Path path_msg;
            path_msg.header.frame_id = "map";
            path_msg.header.stamp = now();
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose.position.x = start_x;
            pose.pose.position.y = start_y;
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
            pub_path_->publish(path_msg);
            pub_waypoints_->publish(path_msg);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                current_path_ = {{{start_x, start_y}}};
            }
            last_plan_time_ = now();
            publishPlannerStatus(planner_type, "no_path", replan_reason, 1,
                                 intersection_detected, planning_ms, cycle_ms_now());
            return;
        }

        // Convert grid indices to world coordinates
        WaypointArray raw_waypoints;
        for (const auto& pt : path_idx) {
            double x, y;
            bridge_.gridToWorld(pt.x, pt.y, x, y);
            raw_waypoints.push_back({{x, y}});
        }

        // Resample to uniform waypoint_spacing in world space
        double wp_spacing = get_parameter("waypoint_spacing").as_double();
        WaypointArray waypoints = resampleWaypointsWorld(raw_waypoints, wp_spacing);

        // If local replan, splice with the tail of the original path
        if (is_local_replan && splice_idx + 1 < current_path.size()) {
            for (size_t i = splice_idx + 1; i < current_path.size(); ++i) {
                waypoints.push_back(current_path[i]);
            }
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

        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_path_ = waypoints;
        }

        last_plan_time_ = now();
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Published %zu waypoints (raw %zu, spacing %.2f m)%s.",
            waypoints.size(), raw_waypoints.size(), wp_spacing, is_local_replan ? " [local]" : "");
        publishPlannerStatus(planner_type, "path_found", replan_reason, waypoints.size(),
                             intersection_detected, planning_ms, cycle_ms_now());
    }

    OccGridBridge bridge_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_live_cloud_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_waypoints_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_occ_grid_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_live_grid_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_planner_status_;
    rclcpp::Publisher<path_planning::msg::AgentCenteredInput>::SharedPtr pub_model_input_;
    rclcpp::Subscription<path_planning::msg::OccupancyGridArray>::SharedPtr sub_model_output_;
    rclcpp::Subscription<path_planning::msg::AgentCenteredInput>::SharedPtr sub_model_output_agent_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    std::mutex mutex_;
    std::vector<std::array<float, 3>> live_points_;
    std::vector<std::array<float, 3>> live_points_staging_;
    bool new_lidar_data_ = false;
    double start_x_ = 0, start_y_ = 0, goal_x_ = 0, goal_y_ = 0;
    double start_z_ = 0;
    double start_roll_ = 0, start_pitch_ = 0, start_yaw_ = 0;
    bool have_pose_ = false, have_goal_ = false;
    bool goal_changed_ = false;
    std::vector<std::array<double, 2>> current_path_;
    rclcpp::Time last_plan_time_{0};
    cv::Mat cached_combined_grid_;
    bool have_cached_combined_grid_ = false;

    path_planning::msg::OccupancyGridArray::SharedPtr last_predicted_array_;
    path_planning::msg::AgentCenteredInput::SharedPtr last_predicted_agent_;
    std::deque<cv::Mat> queue_map_frame_;
    std::deque<EgoFrame> queue_ego_frames_;
    double anchor_x_ = 0, anchor_y_ = 0, anchor_yaw_ = 0;
    cv::Mat pred_only_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PathPlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

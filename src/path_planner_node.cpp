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

#include "path_planning/msg/occupancy_grid_array.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <deque>
#include <mutex>
#include <cstring>
#include <cmath>
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
        declare_parameter<double>("robot_radius", 0.25);
        // Number of RRT* planning iterations (more = better path but slower)
        declare_parameter<int>("rrt_iterations", 10000);
        // RRT* step size in meters (distance to extend tree per iteration, converted to pixels internally)
        declare_parameter<double>("step_size", 0.4);
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

        // Data source: "live" | "map_frame_model" | "agent_centered_model"
        declare_parameter<std::string>("occ_data_mode", "live");
        // Boltzmann temperature for weighting predicted grids (model modes only)
        declare_parameter<double>("prediction_temperature", 1.0);
        // Number of predicted frames we expect from map updater (model modes only)
        declare_parameter<int>("num_predicted_frames", 5);
        // Topic to publish 5 input grids to map updater (model modes only)
        declare_parameter<std::string>("model_occ_input_topic", "/map_updater/occ_grid_input");
        // Topic to subscribe for predicted grids from map updater (model modes only)
        declare_parameter<std::string>("model_predicted_output_topic", "/map_updater/predicted_grid_output");

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

        std::string mode = get_parameter("occ_data_mode").as_string();
        if (mode == "map_frame_model" || mode == "agent_centered_model") {
            std::string input_topic = get_parameter("model_occ_input_topic").as_string();
            std::string output_topic = get_parameter("model_predicted_output_topic").as_string();
            pub_model_input_ = create_publisher<path_planning::msg::OccupancyGridArray>(input_topic, 10);
            sub_model_output_ = create_subscription<path_planning::msg::OccupancyGridArray>(
                output_topic, 10, [this](const path_planning::msg::OccupancyGridArray::SharedPtr msg) {
                    onPredictedGrids(msg);
                });
        }

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
        RCLCPP_INFO(get_logger(), "Received goal pose: x=%.3f, y=%.3f", goal_x_, goal_y_);
    }

    void onPredictedGrids(const path_planning::msg::OccupancyGridArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_predicted_array_ = msg;
    }

    /** Build Boltzmann weights w_i = exp(-i/T) / Z for i = 0..n-1. */
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

    /** Weight-combine N occupancy grids (0-255) into one cv::Mat (0=free, 255=obstacle). */
    static cv::Mat weightCombineGrids(const std::vector<cv::Mat>& grids, const std::vector<double>& weights) {
        if (grids.empty() || weights.size() != grids.size()) return cv::Mat();
        const int h = grids[0].rows, w = grids[0].cols;
        cv::Mat sum = cv::Mat::zeros(h, w, CV_64F);
        for (size_t k = 0; k < grids.size(); ++k) {
            if (grids[k].rows != h || grids[k].cols != w) continue;
            for (int r = 0; r < h; ++r)
                for (int c = 0; c < w; ++c)
                    sum.at<double>(r, c) += weights[k] * static_cast<double>(grids[k].at<uchar>(r, c));
        }
        cv::Mat out(h, w, CV_8UC1);
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                out.at<uchar>(r, c) = sum.at<double>(r, c) >= 127.5 ? 255 : 0;
        return out;
    }

    /** Convert nav_msgs::OccupancyGrid to cv::Mat (0=free, 255=obstacle). */
    static cv::Mat occupancyGridToMat(const nav_msgs::msg::OccupancyGrid& msg) {
        int w = static_cast<int>(msg.info.width);
        int h = static_cast<int>(msg.info.height);
        if (w <= 0 || h <= 0 || msg.data.size() != static_cast<size_t>(w * h)) return cv::Mat();
        cv::Mat m(h, w, CV_8UC1);
        for (int i = 0; i < h * w; ++i) {
            int8_t v = msg.data[i];
            // Handle both standard occupancy values (0-100) and model output (0-1 range stored as 0 or 1)
            // If v is 0 or 1, assume it's from a model that outputs 0-1 range and scale to 0-255
            if (v == 1) {
                m.at<uchar>(i / w, i % w) = 255;
            } else if (v > 1) {
                // Standard occupancy grid format: treat any value > 0 as obstacle
                m.at<uchar>(i / w, i % w) = 255;
            } else {
                m.at<uchar>(i / w, i % w) = 0;
            }
        }
        return m;
    }

    /** Publish a single cv::Mat as OccupancyGrid (0/255 -> 0/100). */
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
        for (int r = 0; r < grid.rows; ++r)
            for (int c = 0; c < grid.cols; ++c)
                occ_msg.data[static_cast<size_t>(r * grid.cols + c)] = grid.at<uchar>(r, c) ? 100 : 0;
        pub_occ_grid_->publish(occ_msg);
    }

    /** Crop full map using sample_* bounds and center on 1216x1216 canvas (unoccupied = 0). */
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

    /** Overlay the crop region from a 1216x1216 canvas onto the full map at sample_* position. */
    void overlayCanvasCropOntoMap(const cv::Mat& canvas_1216,
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
        for (int r = 0; r < crop_h; ++r)
            for (int c = 0; c < crop_w; ++c) {
                uchar v = canvas_1216.at<uchar>(cy + r, cx + c);
                if (v > 0) {
                    int mr = sr_min + r;
                    int mc = sc_min + c;
                    if (mr >= 0 && mr < h && mc >= 0 && mc < w)
                        map_out.at<uchar>(mr, mc) = 255;
                }
            }
    }

    /** Cookie-cutter: zero out the sample_* ROI on the map, then paste the crop from the
     * 1216x1216 canvas into that hole. Planning then uses only the predictions in the ROI. */
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

    /** Build OccupancyGridArray from queue of cv::Mat grids (map frame).
     * Each grid is expected to be MAP_FRAME_CANVAS_SIZE x MAP_FRAME_CANVAS_SIZE (1216x1216).
     * Origin/resolution are set for canvas (0,0 and bridge resolution).
     */
    path_planning::msg::OccupancyGridArray toOccupancyGridArrayMapFrame(
        const std::deque<cv::Mat>& queue) const {
        path_planning::msg::OccupancyGridArray arr;
        arr.header.frame_id = "map";
        arr.header.stamp = now();
        const int canvas_sz = MAP_FRAME_CANVAS_SIZE;
        const float res = static_cast<float>(bridge_.resolution());
        for (const cv::Mat& g : queue) {
            if (g.empty() || g.cols != canvas_sz || g.rows != canvas_sz) continue;
            nav_msgs::msg::OccupancyGrid og;
            og.header = arr.header;
            og.info.resolution = res;
            og.info.width = canvas_sz;
            og.info.height = canvas_sz;
            og.info.origin.position.x = 0.0;
            og.info.origin.position.y = 0.0;
            og.info.origin.position.z = 0.0;
            og.info.origin.orientation.w = 1.0;
            og.data.resize(static_cast<size_t>(canvas_sz * canvas_sz));
            for (int r = 0; r < canvas_sz; ++r)
                for (int c = 0; c < canvas_sz; ++c)
                    og.data[static_cast<size_t>(r * canvas_sz + c)] = g.at<uchar>(r, c) ? 100 : 0;
            arr.grids.push_back(og);
        }
        return arr;
    }

    /** Build OccupancyGridArray from queue of 201x201 ego grids (agent-centered: 5 frames).
     * Now matches save_frame training: 10 grids = occ_0, delta_0, occ_1, delta_1, ..., occ_4, delta_4
     * (delta_0 = zeros). Delta encoded as 0-100: 0=-1, 50=0, 100=+1.
     * Plus motion_forward_speed and motion_yaw_rate (length 5; first element 0).
     */
    path_planning::msg::OccupancyGridArray toOccupancyGridArrayEgo(
        const std::deque<EgoFrame>& queue) const {
        path_planning::msg::OccupancyGridArray arr;
        arr.header.frame_id = "map";
        arr.header.stamp = now();
        const double res = OccGridBridge::egoGridResolution();
        const double ox = OccGridBridge::egoGridOriginX();
        const double oy = OccGridBridge::egoGridOriginY();
        const int sz = OccGridBridge::EGO_GRID_SIZE;

        if (queue.size() != 5u) return arr;

        auto pushGrid = [&](const cv::Mat& g, bool as_occupancy) {
            nav_msgs::msg::OccupancyGrid og;
            og.header = arr.header;
            og.info.resolution = static_cast<float>(res);
            og.info.width = sz;
            og.info.height = sz;
            og.info.origin.position.x = ox;
            og.info.origin.position.y = oy;
            og.info.origin.position.z = 0.0;
            og.info.origin.orientation.w = 1.0;
            og.data.resize(static_cast<size_t>(sz * sz));
            for (int r = 0; r < sz; ++r)
                for (int c = 0; c < sz; ++c) {
                    int v = as_occupancy ? (g.at<uchar>(r, c) ? 100 : 0) : static_cast<int>(g.at<uchar>(r, c));
                    og.data[static_cast<size_t>(r * sz + c)] = static_cast<int8_t>(std::max(0, std::min(100, v)));
                }
            arr.grids.push_back(og);
        };

        std::vector<cv::Mat> occ_float(5);
        for (size_t i = 0; i < 5u; ++i) {
            occ_float[i] = cv::Mat(sz, sz, CV_32FC1);
            for (int r = 0; r < sz; ++r)
                for (int c = 0; c < sz; ++c)
                    occ_float[i].at<float>(r, c) = queue[i].grid.at<uchar>(r, c) ? 1.f : 0.f;
        }

        for (size_t i = 0; i < 5u; ++i) {
            pushGrid(queue[i].grid, true);
            cv::Mat delta_enc(sz, sz, CV_8UC1);
            if (i == 0) {
                delta_enc.setTo(50);
            } else {
                for (int r = 0; r < sz; ++r)
                    for (int c = 0; c < sz; ++c) {
                        float d = occ_float[i].at<float>(r, c) - occ_float[i - 1].at<float>(r, c);
                        int v = static_cast<int>(std::round((d + 1.f) * 50.f));
                        delta_enc.at<uchar>(r, c) = static_cast<uchar>(std::max(0, std::min(100, v)));
                    }
            }
            pushGrid(delta_enc, false);
        }

        arr.motion_forward_speed.resize(5);
        arr.motion_yaw_rate.resize(5);
        arr.motion_forward_speed[0] = 0.f;
        arr.motion_yaw_rate[0] = 0.f;
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
            arr.motion_forward_speed[j] = static_cast<float>(body_x / dt);
            arr.motion_yaw_rate[j] = static_cast<float>(dyaw / dt);
        }
        return arr;
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

        const double resolution = bridge_.resolution();
        double cumulative_dist = 0.0;

        const Eigen::Vector2d robot(robot_x, robot_y);

        // Find closest point on path to robot (start checking from there)
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

        // Check path segments starting from closest point
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

                if (col >= 0 && col < occupancy_grid.cols && row >= 0 && row < occupancy_grid.rows) {
                    if (occupancy_grid.at<uchar>(row, col) != 0) return true;
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
        std::vector<std::array<float, 3>> out;
        out.reserve(points_lidar.size());

        // Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll) using Eigen
        Eigen::AngleAxisd roll_angle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R = (yaw_angle * pitch_angle * roll_angle).toRotationMatrix();
        Eigen::Vector3d t(tx, ty, tz);

        for (const auto& pt : points_lidar) {
            Eigen::Vector3d p(static_cast<double>(pt[0]), static_cast<double>(pt[1]), static_cast<double>(pt[2]));
            Eigen::Vector3d p_m = R * p + t;
            out.push_back({{static_cast<float>(p_m.x()), static_cast<float>(p_m.y()), static_cast<float>(p_m.z())}});
        }
        return out;
    }

    void runPlanningCycle() {
        if (!bridge_.isInitialized()) return;

        std::vector<std::array<float, 3>> live_pts;
        double start_x, start_y, goal_x, goal_y;
        double start_z, start_roll, start_pitch, start_yaw;
        bool have_pose, have_goal;
        std::vector<std::array<double, 2>> current_path;
        path_planning::msg::OccupancyGridArray::SharedPtr predicted_msg;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            live_pts = live_points_;
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
        }

        std::string mode = get_parameter("occ_data_mode").as_string();
        double z_min = get_parameter("z_min").as_double();
        double z_max = get_parameter("z_max").as_double();

        if (have_pose) {
            live_pts = transformPointsToMap(live_pts, start_x, start_y, start_z,
                                            start_roll, start_pitch, start_yaw);
        } else {
            live_pts.clear();
        }

        cv::Mat combined;
        if (mode == "live") {
            cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
            combined = bridge_.mergeWithStaticMap(live_grid);
        } else if (mode == "map_frame_model") {
            cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
            cv::Mat current_combined = bridge_.mergeWithStaticMap(live_grid);
            int sample_col_min = get_parameter("sample_col_min").as_int();
            int sample_col_max = get_parameter("sample_col_max").as_int();
            int sample_row_min = get_parameter("sample_row_min").as_int();
            int sample_row_max = get_parameter("sample_row_max").as_int();
            if (have_pose) {
                cv::Mat canvas = cropMapAndCenterOnCanvas(
                    current_combined, sample_col_min, sample_col_max, sample_row_min, sample_row_max);
                if (!canvas.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    queue_map_frame_.push_back(canvas);
                    while (queue_map_frame_.size() > 5u) queue_map_frame_.pop_front();
                }
            }
            if (pub_model_input_) {
                std::deque<cv::Mat> q;
                { std::lock_guard<std::mutex> lock(mutex_); q = queue_map_frame_; }
                if (q.size() == 5u) {
                    path_planning::msg::OccupancyGridArray arr = toOccupancyGridArrayMapFrame(q);
                    pub_model_input_->publish(arr);
                }
            }
            if (predicted_msg && !predicted_msg->grids.empty()) {
                int N = get_parameter("num_predicted_frames").as_int();
                double T = get_parameter("prediction_temperature").as_double();
                size_t n = std::min(static_cast<size_t>(N), predicted_msg->grids.size());
                std::vector<cv::Mat> grids(n);
                for (size_t i = 0; i < n; ++i)
                    grids[i] = occupancyGridToMat(predicted_msg->grids[i]);
                std::vector<double> w = boltmannWeights(static_cast<int>(n), T);
                cv::Mat weighted = weightCombineGrids(grids, w);
                if (!weighted.empty()) {
                    int sc_min = get_parameter("sample_col_min").as_int();
                    int sc_max = get_parameter("sample_col_max").as_int();
                    int sr_min = get_parameter("sample_row_min").as_int();
                    int sr_max = get_parameter("sample_row_max").as_int();
                    combined = bridge_.mergeWithStaticMap(
                        cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1));
                    cookieCutterCanvasCropOntoMap(weighted, sc_min, sc_max, sr_min, sr_max, combined);
                } else {
                    combined = current_combined;
                }
            } else {
                combined = current_combined;
            }
        } else if (mode == "agent_centered_model") {
            if (have_pose) {
                cv::Mat ego_grid = OccGridBridge::pointcloudToEgoOccupancyGrid201(
                    live_pts, start_x, start_y, start_yaw, z_min, z_max);
                EgoFrame frame;
                frame.grid = ego_grid;
                frame.x = start_x;
                frame.y = start_y;
                frame.yaw = start_yaw;
                frame.time = now();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    queue_ego_frames_.push_back(frame);
                    while (queue_ego_frames_.size() > 5u) queue_ego_frames_.pop_front();
                    anchor_x_ = queue_ego_frames_.back().x;
                    anchor_y_ = queue_ego_frames_.back().y;
                    anchor_yaw_ = queue_ego_frames_.back().yaw;
                }
                if (pub_model_input_ && queue_ego_frames_.size() == 5u) {
                    std::deque<EgoFrame> q;
                    { std::lock_guard<std::mutex> lock(mutex_); q = queue_ego_frames_; }
                    path_planning::msg::OccupancyGridArray arr = toOccupancyGridArrayEgo(q);
                    pub_model_input_->publish(arr);
                }
            }
            if (predicted_msg && !predicted_msg->grids.empty()) {
                int N = get_parameter("num_predicted_frames").as_int();
                double T = get_parameter("prediction_temperature").as_double();
                size_t n = std::min(static_cast<size_t>(N), predicted_msg->grids.size());
                std::vector<cv::Mat> grids(n);
                for (size_t i = 0; i < n; ++i)
                    grids[i] = occupancyGridToMat(predicted_msg->grids[i]);
                std::vector<double> w = boltmannWeights(static_cast<int>(n), T);
                cv::Mat weighted_ego = weightCombineGrids(grids, w);
                if (!weighted_ego.empty()) {
                    double ax, ay, ayaw;
                    { std::lock_guard<std::mutex> lock(mutex_); ax = anchor_x_; ay = anchor_y_; ayaw = anchor_yaw_; }
                    combined = bridge_.mergeWithStaticMap(cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1));
                    bridge_.zeroEgoFootprintInMap(ax, ay, ayaw, combined);
                    bridge_.pasteEgoGridIntoMap(weighted_ego, ax, ay, ayaw, combined);
                } else {
                    combined = bridge_.mergeWithStaticMap(cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1));
                }
            } else {
                combined = bridge_.mergeWithStaticMap(cv::Mat::zeros(bridge_.gridHeight(), bridge_.gridWidth(), CV_8UC1));
            }
        } else {
            cv::Mat live_grid = bridge_.pointcloudToOccupancyGrid(live_pts, z_min, z_max);
            combined = bridge_.mergeWithStaticMap(live_grid);
        }

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Combined grid: %d x %d (mode=%s)", combined.cols, combined.rows, mode.c_str());

        // Publish current combined occupancy grid for visualization
        publishOccupancyGrid(combined, "map", bridge_.xMin(), bridge_.yMin(), bridge_.resolution());

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
        double step_size_m = get_parameter("step_size").as_double();
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
    rclcpp::Publisher<path_planning::msg::OccupancyGridArray>::SharedPtr pub_model_input_;
    rclcpp::Subscription<path_planning::msg::OccupancyGridArray>::SharedPtr sub_model_output_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    std::mutex mutex_;
    std::vector<std::array<float, 3>> live_points_;
    double start_x_ = 0, start_y_ = 0, goal_x_ = 0, goal_y_ = 0;
    double start_z_ = 0;
    double start_roll_ = 0, start_pitch_ = 0, start_yaw_ = 0;
    bool have_pose_ = false, have_goal_ = false;
    std::vector<std::array<double, 2>> current_path_;
    rclcpp::Time last_plan_time_{0};

    // Model modes: queues and last predicted message
    std::deque<cv::Mat> queue_map_frame_;
    std::deque<EgoFrame> queue_ego_frames_;
    path_planning::msg::OccupancyGridArray::SharedPtr last_predicted_array_;
    double anchor_x_ = 0, anchor_y_ = 0, anchor_yaw_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PathPlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

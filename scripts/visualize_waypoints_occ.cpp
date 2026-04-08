#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/parameter_value.hpp>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <mutex>
#include <vector>
#include <optional>
#include <cmath>
#include <sstream>
#include <iostream>
#include <cstdio>

enum class ViewMode { FULL, ROI, FOLLOW };

static constexpr int DRAG_THRESHOLD_PX = 5;

struct OccData {
    int width = 0;
    int height = 0;
    double resolution = 0.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    std::vector<int8_t> data;
};

class WaypointsOccVisualizer : public rclcpp::Node {
public:
    WaypointsOccVisualizer()
        : Node("waypoints_occ_visualizer") {
        std::string occ_topic = declare_parameter<std::string>("occ_topic", "/occupancy_grid");
        std::string path_topic = declare_parameter<std::string>("path_topic", "/planned_path");
        std::string waypoints_topic = declare_parameter<std::string>("waypoints_topic", "/waypoints");
        std::string pose_topic = declare_parameter<std::string>("pose_topic", "/pcl_pose");
        std::string goal_topic = declare_parameter<std::string>("goal_topic", "/move_base_simple/goal");
        std::string planner_status_topic = declare_parameter<std::string>("planner_status_topic", "/planner_status");
        rate_hz_ = declare_parameter<double>("rate", 10.0);
        view_col_min_ = declare_parameter<int>("view_col_min", -1);
        view_col_max_ = declare_parameter<int>("view_col_max", -1);
        view_row_min_ = declare_parameter<int>("view_row_min", -1);
        view_row_max_ = declare_parameter<int>("view_row_max", -1);
        show_energy_map_ = declare_parameter<bool>("show_energy_map", true);
        show_agent_centered_roi_ = declare_parameter<bool>("show_agent_centered_roi", true);
        show_robot_marker_ = declare_parameter<bool>("show_robot_marker", true);
        show_goal_marker_ = declare_parameter<bool>("show_goal_marker", true);
        follow_radius_m_ = declare_parameter<double>("follow_radius_m", 5.0);

        rclcpp::QoS qos_be(1);
        qos_be.best_effort().durability_volatile();

        sub_occ_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            occ_topic, qos_be, [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                onOcc(msg);
            });
        sub_path_ = create_subscription<nav_msgs::msg::Path>(
            path_topic, qos_be, [this](const nav_msgs::msg::Path::SharedPtr msg) {
                onPath(msg, path_poses_);
            });
        sub_waypoints_ = create_subscription<nav_msgs::msg::Path>(
            waypoints_topic, qos_be, [this](const nav_msgs::msg::Path::SharedPtr msg) {
                onPath(msg, waypoint_poses_);
            });
        if (!pose_topic.empty()) {
            sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                pose_topic, qos_be,
                [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                    onPose(msg);
                });
        }
        if (!goal_topic.empty()) {
            sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
                goal_topic, qos_be,
                [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                    onGoal(msg);
                });
            pub_goal_ = create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic, qos_be);
        }
        if (!planner_status_topic.empty()) {
            sub_planner_status_ = create_subscription<std_msgs::msg::String>(
                planner_status_topic, qos_be,
                [this](const std_msgs::msg::String::SharedPtr msg) {
                    onPlannerStatus(msg);
                });
        }

        timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate_hz_)),
            [this]() { render(); });

        // Flush stdout before creating window to prevent GTK initialization issues
        // when output is piped (the "| grep err" workaround symptom)
        std::cout << std::flush;
        std::fflush(stdout);
        std::fflush(stderr);

        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, 900, 900);
        cv::setMouseCallback(window_name_, &WaypointsOccVisualizer::onMouse, this);
        RCLCPP_INFO(get_logger(),
            "Click=drag region to zoom | R=reset full view | F=toggle follow robot | Left-click=set goal");
        door_toggle_enabled_ = declare_parameter<bool>("door_toggle_enabled", false);
        if (door_toggle_enabled_) {
            pub_door_toggle_ = create_publisher<std_msgs::msg::String>("/door_toggle", qos_be);
        }

        RCLCPP_INFO(get_logger(),
            "M=toggle model overlay | W/S=corridor width +/-0.01m%s",
            door_toggle_enabled_ ? " | O=toggle door map" : "");

        param_client_ = create_client<rcl_interfaces::srv::SetParameters>("/path_planner/set_parameters");
    }

private:
    void setRemoteParam(const std::string& name, bool value) {
        if (!param_client_ || !param_client_->service_is_ready()) return;
        auto req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();
        rcl_interfaces::msg::Parameter p;
        p.name = name;
        p.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
        p.value.bool_value = value;
        req->parameters.push_back(p);
        param_client_->async_send_request(req);
    }

    void setRemoteParam(const std::string& name, double value) {
        if (!param_client_ || !param_client_->service_is_ready()) return;
        auto req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();
        rcl_interfaces::msg::Parameter p;
        p.name = name;
        p.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
        p.value.double_value = value;
        req->parameters.push_back(p);
        param_client_->async_send_request(req);
    }

    void onOcc(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        occ_.width = static_cast<int>(msg->info.width);
        occ_.height = static_cast<int>(msg->info.height);
        occ_.resolution = msg->info.resolution;
        occ_.origin_x = msg->info.origin.position.x;
        occ_.origin_y = msg->info.origin.position.y;
        occ_.data.assign(msg->data.begin(), msg->data.end());
    }

    void onPath(const nav_msgs::msg::Path::SharedPtr msg,
                std::vector<cv::Point2d>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        out.clear();
        out.reserve(msg->poses.size());
        for (const auto& p : msg->poses) {
            out.emplace_back(p.pose.position.x, p.pose.position.y);
        }
    }

    void onPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        robot_pose_ = cv::Point2d(msg->pose.pose.position.x, msg->pose.pose.position.y);
    }

    void onGoal(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        goal_pose_ = cv::Point2d(msg->pose.position.x, msg->pose.position.y);
    }

    void onPlannerStatus(const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        planner_status_text_ = msg->data;
    }

    bool worldToPixel(const OccData& occ, const cv::Point2d& w, cv::Point& p) const {
        if (occ.width <= 0 || occ.height <= 0 || occ.resolution <= 0.0) return false;
        Eigen::Vector2d world(w.x, w.y);
        // Top-left of cell (0,0): origin is bottom-left, so top = origin_y + (height-1)*res (matches OccGridBridge)
        Eigen::Vector2d origin(occ.origin_x, occ.origin_y + (occ.height - 1) * occ.resolution);
        Eigen::Vector2d col_row = (world - origin).array() / occ.resolution;
        col_row.y() = -col_row.y();
        int c = static_cast<int>(std::round(col_row.x()));
        int r = static_cast<int>(std::round(col_row.y()));
        if (c < 0 || c >= occ.width || r < 0 || r >= occ.height) return false;
        p = cv::Point(c, r);
        return true;
    }

    cv::Point2d pixelToWorld(const OccData& occ, int col, int row) const {
        Eigen::Vector2d p(
            occ.origin_x + col * occ.resolution,
            occ.origin_y + (occ.height - 1 - row) * occ.resolution);
        return cv::Point2d(p.x(), p.y());
    }

    static void onMouse(int event, int x, int y, int, void* userdata) {
        auto* self = static_cast<WaypointsOccVisualizer*>(userdata);
        std::lock_guard<std::mutex> lock(self->mouse_mutex_);

        // Convert from display (window) space to original image space.
        // With WINDOW_NORMAL, mouse coords are in window pixels which may differ
        // from image pixels when the image is scaled for display.
        double s = std::max(1.0, self->display_scale_);
        int ix = static_cast<int>(x / s);
        int iy = static_cast<int>(y / s);

        if (event == cv::EVENT_LBUTTONDOWN) {
            self->drag_start_x_ = ix;
            self->drag_start_y_ = iy;
            self->drag_end_x_ = ix;
            self->drag_end_y_ = iy;
            self->is_dragging_ = true;
            return;
        }

        if (event == cv::EVENT_LBUTTONUP) {
            if (self->is_dragging_) {
                int dx = ix - self->drag_start_x_;
                int dy = iy - self->drag_start_y_;
                int dist_sq = dx * dx + dy * dy;
                int thresh = std::max(2, static_cast<int>(DRAG_THRESHOLD_PX / s));
                if (dist_sq >= thresh * thresh) {
                    int pw = self->last_occ_panel_width_;
                    int ph = self->last_occ_panel_height_;
                    if (pw > 0 && ph > 0 &&
                        self->drag_start_x_ >= 0 && self->drag_start_x_ < pw &&
                        self->drag_start_y_ >= 0 && self->drag_start_y_ < ph &&
                        ix >= 0 && ix < pw && iy >= 0 && iy < ph) {
                        int x0 = std::min(self->drag_start_x_, ix);
                        int x1 = std::max(self->drag_start_x_, ix);
                        int y0 = std::min(self->drag_start_y_, iy);
                        int y1 = std::max(self->drag_start_y_, iy);
                        int w = std::max(2, x1 - x0);
                        int h = std::max(2, y1 - y0);
                        self->view_col_min_ = self->last_col_min_ + x0;
                        self->view_row_min_ = self->last_row_min_ + y0;
                        self->view_col_max_ = self->last_col_min_ + x0 + w - 1;
                        self->view_row_max_ = self->last_row_min_ + y0 + h - 1;
                        self->view_mode_ = ViewMode::ROI;
                        self->follow_robot_ = false;
                    }
                } else {
                    self->click_x_ = self->drag_start_x_;
                    self->click_y_ = self->drag_start_y_;
                    self->has_click_ = true;
                }
            }
            self->is_dragging_ = false;
            return;
        }

        if (event == cv::EVENT_MOUSEMOVE) {
            if (self->is_dragging_) {
                self->drag_end_x_ = ix;
                self->drag_end_y_ = iy;
            }
            self->mouse_x_ = ix;
            self->mouse_y_ = iy;
            self->has_mouse_ = true;
        }
    }

    void render() {
        OccData occ;
        std::vector<cv::Point2d> path;
        std::vector<cv::Point2d> waypoints;
        std::optional<cv::Point2d> robot;
        std::optional<cv::Point2d> goal;
        std::string planner_status;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            occ = occ_;
            path = path_poses_;
            waypoints = waypoint_poses_;
            robot = robot_pose_;
            goal = goal_pose_;
            planner_status = planner_status_text_;
        }

        if (occ.width == 0 || occ.height == 0 || occ.data.empty()) {
            return;
        }

        cv::Mat img(occ.height, occ.width, CV_8UC1);
        const int8_t* src = occ.data.data();
        for (int r = 0; r < occ.height; ++r) {
            uchar* dst = img.ptr<uchar>(r);
            for (int c = 0; c < occ.width; ++c) {
                int8_t val = src[r * occ.width + c];
                if (val < 0) {
                    dst[c] = 127;
                } else if (val <= 100) {
                    int v = 100 - val;
                    v = std::max(0, std::min(100, v));
                    dst[c] = static_cast<uchar>(v * 255 / 100);
                } else {
                    dst[c] = 0;
                }
            }
        }

        cv::Mat vis;
        cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);

        for (int r = 0; r < occ.height; ++r) {
            for (int c = 0; c < occ.width; ++c) {
                if (src[r * occ.width + c] == 101)
                    vis.at<cv::Vec3b>(r, c) = cv::Vec3b(180, 60, 0);
            }
        }

        // Optional: build an "energy" map that matches occupancy_grid_planning_energy_opt.py:
        // distance transform of free space visualized with a viridis-style colormap.
        cv::Mat energy_color;
        if (show_energy_map_) {
            cv::Mat binary_inv(occ.height, occ.width, CV_8UC1);
            const int8_t* src = occ.data.data();
            for (int r = 0; r < occ.height; ++r) {
                uchar* dst = binary_inv.ptr<uchar>(r);
                for (int c = 0; c < occ.width; ++c) {
                    int8_t val = src[r * occ.width + c];
                    bool free_cell = (val >= 0 && val <= 100 && val < 50);
                    dst[c] = free_cell ? 255 : 0;
                }
            }
            cv::Mat dist;
            cv::distanceTransform(binary_inv, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

            double min_val = 0.0, max_val = 0.0;
            cv::minMaxLoc(dist, &min_val, &max_val);

            cv::Mat dist_norm_u8;
            if (max_val > 0.0) {
                // Scale so that the maximum clearance uses the top of the colormap,
                // similar to matplotlib imshow(planner.dist, cmap="viridis").
                dist.convertTo(dist_norm_u8, CV_8UC1, 255.0 / max_val);
            } else {
                dist_norm_u8 = cv::Mat::zeros(occ.height, occ.width, CV_8UC1);
            }

            energy_color = cv::Mat(occ.height, occ.width, CV_8UC3);
            cv::applyColorMap(dist_norm_u8, energy_color, cv::COLORMAP_TURBO);
        }

        int col_min = 0;
        int col_max = occ.width - 1;
        int row_min = 0;
        int row_max = occ.height - 1;
        if (follow_robot_ && robot.has_value()) {
            double res = occ.resolution;
            double rx = robot->x, ry = robot->y;
            double half = follow_radius_m_;
            double wx_lo = rx - half, wx_hi = rx + half;
            double wy_lo = ry - half, wy_hi = ry + half;
            double ox = occ.origin_x, oy = occ.origin_y;
            int h = occ.height, w = occ.width;
            col_min = std::max(0, static_cast<int>(std::floor((wx_lo - ox) / res)));
            col_max = std::min(w - 1, static_cast<int>(std::ceil((wx_hi - ox) / res)));
            row_min = std::max(0, static_cast<int>(std::floor((oy + (h - 1) * res - wy_hi) / res)));
            row_max = std::min(h - 1, static_cast<int>(std::ceil((oy + (h - 1) * res - wy_lo) / res)));
            if (col_max < col_min) std::swap(col_min, col_max);
            if (row_max < row_min) std::swap(row_min, row_max);
        } else if (view_mode_ == ViewMode::ROI && view_col_min_ >= 0) {
            col_min = view_col_min_;
            col_max = view_col_max_;
            row_min = view_row_min_;
            row_max = view_row_max_;
        }

        col_min = std::max(0, std::min(col_min, occ.width - 1));
        col_max = std::max(0, std::min(col_max, occ.width - 1));
        row_min = std::max(0, std::min(row_min, occ.height - 1));
        row_max = std::max(0, std::min(row_max, occ.height - 1));
        if (col_max < col_min) std::swap(col_min, col_max);
        if (row_max < row_min) std::swap(row_min, row_max);

        cv::Rect roi(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1);
        cv::Mat view_occ = vis(roi);

        {
            std::lock_guard<std::mutex> lock(mouse_mutex_);
            last_col_min_ = col_min;
            last_row_min_ = row_min;
            last_occ_panel_width_ = view_occ.cols;
            last_occ_panel_height_ = view_occ.rows;
        }
        cv::Mat view_energy;
        if (show_energy_map_ && !energy_color.empty()) {
            view_energy = energy_color(roi);
        }

        auto draw_polyline = [&](const std::vector<cv::Point2d>& pts,
                                 const cv::Scalar& color,
                                 cv::Mat& canvas) {
            if (pts.size() < 2) return;
            std::vector<cv::Point> px;
            for (const auto& w : pts) {
                cv::Point p;
                if (worldToPixel(occ, w, p)) px.push_back(p - cv::Point(col_min, row_min));
            }
            for (size_t i = 1; i < px.size(); ++i) {
                cv::line(canvas, px[i - 1], px[i], color, 2);
            }
        };

        // Draw path (blue) on occupancy view (and energy view if enabled)
        draw_polyline(path, cv::Scalar(255, 0, 0), view_occ);
        if (show_energy_map_ && !view_energy.empty()) {
            draw_polyline(path, cv::Scalar(255, 0, 0), view_energy);
        }

        // Waypoints (red dots), robot (green), goal (yellow star)
        for (const auto& w : waypoints) {
            cv::Point p;
            if (worldToPixel(occ, w, p)) {
                cv::Point p_local = p - cv::Point(col_min, row_min);
                cv::circle(view_occ, p_local, 3, cv::Scalar(0, 0, 255), -1);
                if (show_energy_map_ && !view_energy.empty()) {
                    cv::circle(view_energy, p_local, 3, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        if (show_robot_marker_ && robot.has_value()) {
            cv::Point p;
            if (worldToPixel(occ, robot.value(), p)) {
                cv::Point p_local = p - cv::Point(col_min, row_min);
                cv::circle(view_occ, p_local, 6, cv::Scalar(0, 255, 0), -1);
                if (show_energy_map_ && !view_energy.empty()) {
                    cv::circle(view_energy, p_local, 6, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        if (show_goal_marker_ && goal.has_value()) {
            cv::Point p;
            if (worldToPixel(occ, goal.value(), p)) {
                cv::Point p_local = p - cv::Point(col_min, row_min);
                cv::drawMarker(view_occ, p_local,
                               cv::Scalar(0, 255, 255), cv::MARKER_STAR, 12, 2);
                if (show_energy_map_ && !view_energy.empty()) {
                    cv::drawMarker(view_energy, p_local,
                                   cv::Scalar(0, 255, 255), cv::MARKER_STAR, 12, 2);
                }
            }
        }

        // Handle click to set goal
        int click_x = -1;
        int click_y = -1;
        {
            std::lock_guard<std::mutex> lock(mouse_mutex_);
            if (has_click_) {
                click_x = click_x_;
                click_y = click_y_;
                has_click_ = false;
            }
        }
        int occ_width = view_occ.cols;
        int occ_height = view_occ.rows;

        if (click_x >= 0 && click_y >= 0 &&
            click_x < occ_width && click_y < occ_height && pub_goal_) {
            int col = col_min + click_x;
            int row = row_min + click_y;
            cv::Point2d world_pos = pixelToWorld(occ, col, row);
            
            auto goal_msg = geometry_msgs::msg::PoseStamped();
            goal_msg.header.stamp = now();
            goal_msg.header.frame_id = "map";
            goal_msg.pose.position.x = world_pos.x;
            goal_msg.pose.position.y = world_pos.y;
            goal_msg.pose.position.z = 0.0;
            goal_msg.pose.orientation.w = 1.0;
            
            pub_goal_->publish(goal_msg);
            RCLCPP_INFO(get_logger(), "Goal set to (%.2f, %.2f)", world_pos.x, world_pos.y);
            
            // Update local goal for visualization
            std::lock_guard<std::mutex> lock(mutex_);
            goal_pose_ = world_pos;
        }

        int mouse_x = -1;
        int mouse_y = -1;
        {
            std::lock_guard<std::mutex> lock(mouse_mutex_);
            if (has_mouse_) {
                mouse_x = mouse_x_;
                mouse_y = mouse_y_;
            }
        }
        std::string mode_str = follow_robot_ ? "FOLLOW" : (view_mode_ == ViewMode::ROI ? "ZOOM" : "FULL");
        std::string mouse_cr = "Mouse col,row: out of bounds";
        std::string mouse_xy = "Mouse x,y: out of bounds";
        if (mouse_x >= 0 && mouse_y >= 0 &&
            mouse_x < occ_width && mouse_y < occ_height) {
            int col = col_min + mouse_x;
            int row = row_min + mouse_y;
            cv::Point2d wp = pixelToWorld(occ, col, row);
            mouse_cr = "Mouse col,row: " + std::to_string(col) + ", " + std::to_string(row);
            mouse_xy = "Mouse x,y: " + cv::format("%.2f, %.2f", wp.x, wp.y);
        }

        // Draw drag selection rectangle while dragging
        {
            std::lock_guard<std::mutex> lock(mouse_mutex_);
            if (is_dragging_ && last_occ_panel_width_ > 0 && last_occ_panel_height_ > 0) {
                int x0 = std::max(0, std::min(drag_start_x_, drag_end_x_));
                int x1 = std::min(occ_width - 1, std::max(drag_start_x_, drag_end_x_));
                int y0 = std::max(0, std::min(drag_start_y_, drag_end_y_));
                int y1 = std::min(occ_height - 1, std::max(drag_start_y_, drag_end_y_));
                if (x0 < occ_width && y0 < occ_height && x1 >= x0 && y1 >= y0) {
                    cv::rectangle(view_occ, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 255), 2);
                }
            }
        }

        // Build single combined view: occupancy on the left, energy map and/or agent-centered ROI as needed
        cv::Mat combined = view_occ.clone();
        if (show_energy_map_ && !view_energy.empty()) {
            cv::Mat temp;
            cv::hconcat(combined, view_energy, temp);
            combined = temp;
        }

        if (show_agent_centered_roi_ && robot.has_value()) {
            const double agent_radius_m = 5.0;
            double rx = robot->x;
            double ry = robot->y;
            double wx_lo = rx - agent_radius_m;
            double wx_hi = rx + agent_radius_m;
            double wy_lo = ry - agent_radius_m;
            double wy_hi = ry + agent_radius_m;
            double ox = occ.origin_x;
            double oy = occ.origin_y;
            double res = occ.resolution;
            int h = occ.height;
            int w = occ.width;
            int ac_col_min = static_cast<int>(std::floor((wx_lo - ox) / res));
            int ac_col_max = static_cast<int>(std::ceil((wx_hi - ox) / res));
            int ac_row_min = static_cast<int>(std::floor((oy + (h - 1) * res - wy_hi) / res));
            int ac_row_max = static_cast<int>(std::ceil((oy + (h - 1) * res - wy_lo) / res));
            ac_col_min = std::max(0, std::min(ac_col_min, w - 1));
            ac_col_max = std::max(0, std::min(ac_col_max, w - 1));
            ac_row_min = std::max(0, std::min(ac_row_min, h - 1));
            ac_row_max = std::max(0, std::min(ac_row_max, h - 1));
            if (ac_col_max < ac_col_min) std::swap(ac_col_min, ac_col_max);
            if (ac_row_max < ac_row_min) std::swap(ac_row_min, ac_row_max);

            cv::Rect ac_roi(ac_col_min, ac_row_min, ac_col_max - ac_col_min + 1, ac_row_max - ac_row_min + 1);
            if (ac_roi.width > 0 && ac_roi.height > 0) {
                cv::Mat view_agent = vis(ac_roi).clone();
                cv::Point ac_offset(ac_col_min, ac_row_min);

                auto draw_polyline_agent = [&](const std::vector<cv::Point2d>& pts, const cv::Scalar& color) {
                    if (pts.size() < 2) return;
                    for (size_t i = 1; i < pts.size(); ++i) {
                        cv::Point p0, p1;
                        if (worldToPixel(occ, pts[i - 1], p0) && worldToPixel(occ, pts[i], p1)) {
                            cv::Point q0 = p0 - ac_offset;
                            cv::Point q1 = p1 - ac_offset;
                            if (q0.x >= 0 && q0.x < view_agent.cols && q0.y >= 0 && q0.y < view_agent.rows &&
                                q1.x >= 0 && q1.x < view_agent.cols && q1.y >= 0 && q1.y < view_agent.rows)
                                cv::line(view_agent, q0, q1, color, 2);
                        }
                    }
                };
                draw_polyline_agent(path, cv::Scalar(255, 0, 0));
                for (const auto& wp : waypoints) {
                    cv::Point p;
                    if (worldToPixel(occ, wp, p)) {
                        cv::Point p_local = p - ac_offset;
                        if (p_local.x >= 0 && p_local.x < view_agent.cols && p_local.y >= 0 && p_local.y < view_agent.rows)
                            cv::circle(view_agent, p_local, 3, cv::Scalar(0, 0, 255), -1);
                    }
                }
                if (show_robot_marker_ && robot.has_value()) {
                    cv::Point p;
                    if (worldToPixel(occ, robot.value(), p)) {
                        cv::Point p_local = p - ac_offset;
                        if (p_local.x >= 0 && p_local.x < view_agent.cols && p_local.y >= 0 && p_local.y < view_agent.rows)
                            cv::circle(view_agent, p_local, 6, cv::Scalar(0, 255, 0), -1);
                    }
                }
                if (show_goal_marker_ && goal.has_value()) {
                    cv::Point p;
                    if (worldToPixel(occ, goal.value(), p)) {
                        cv::Point p_local = p - ac_offset;
                        if (p_local.x >= 0 && p_local.x < view_agent.cols && p_local.y >= 0 && p_local.y < view_agent.rows)
                            cv::drawMarker(view_agent, p_local, cv::Scalar(0, 255, 255), cv::MARKER_STAR, 12, 2);
                    }
                }
                int baseline = 0;
                cv::Size text_size = cv::getTextSize("Agent 5m", cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(view_agent, cv::Rect(2, 2, text_size.width + 6, text_size.height + baseline + 6),
                              cv::Scalar(255, 255, 255), cv::FILLED);
                cv::putText(view_agent, "Agent 5m", cv::Point(5, 5 + text_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                cv::Mat temp;
                cv::hconcat(combined, view_agent, temp);
                combined = temp;
            }
        }

        // Scale occupancy visualization up if the ROI is small, so the window stays usable and
        // mouse coordinates map reliably to occupancy image pixels.
        double scale = 1.0;
        if (combined.rows < 600) {
            scale = 600.0 / combined.rows;
        }
        cv::Mat display_occ;
        if (scale > 1.01) {
            cv::resize(combined, display_occ, cv::Size(), scale, scale, cv::INTER_NEAREST);
        } else {
            display_occ = combined;
        }

        // Build fixed-size side panel so text stays a constant on-screen size.
        const int side_w = 490;
        cv::Mat side_panel(display_occ.rows, side_w, CV_8UC3, cv::Scalar(245, 245, 245));
        cv::line(side_panel, cv::Point(0, 0), cv::Point(0, side_panel.rows - 1), cv::Scalar(180, 180, 180), 1);

        int y = 22;
        const int line_h = 22;
        const double font = 0.5;
        auto put_line = [&](const std::string& text, const cv::Scalar& color = cv::Scalar(20, 20, 20)) {
            if (y < side_panel.rows - 8) {
                cv::putText(side_panel, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, font, color, 1, cv::LINE_AA);
            }
            y += line_h;
        };

        put_line("View mode: " + mode_str, cv::Scalar(40, 40, 120));
        put_line(mouse_cr);
        put_line(mouse_xy);
        y += 8;
        if (robot.has_value()) {
            put_line("Robot x,y: " + cv::format("%.2f, %.2f", robot->x, robot->y), cv::Scalar(0, 120, 0));
        } else {
            put_line("Robot x,y: unavailable");
        }
        if (goal.has_value()) {
            put_line("Goal x,y: " + cv::format("%.2f, %.2f", goal->x, goal->y), cv::Scalar(0, 120, 120));
        } else {
            put_line("Goal x,y: unavailable");
        }
        y += 8;
        put_line("Planner status:", cv::Scalar(120, 40, 40));
        if (!planner_status.empty()) {
            std::istringstream ss(planner_status);
            std::string line;
            while (std::getline(ss, line)) {
                if (!line.empty()) put_line(line);
            }
        } else {
            put_line("No planner status yet");
        }
        y += 8;
        put_line("--- Live Controls ---", cv::Scalar(120, 0, 120));
        put_line("M: Model overlay " + std::string(use_model_overlay_ ? "ON" : "OFF"),
                 use_model_overlay_ ? cv::Scalar(0, 120, 0) : cv::Scalar(0, 0, 180));
        put_line(cv::format("W/S: Corridor width %.2f m", corridor_width_cm_ * 0.01));
        if (door_toggle_enabled_) {
            put_line("O: Door map " + std::to_string(door_map_index_),
                     door_map_index_ == 1 ? cv::Scalar(0, 120, 0) : cv::Scalar(180, 0, 0));
        }

        cv::Mat display;
        cv::hconcat(display_occ, side_panel, display);
        {
            std::lock_guard<std::mutex> lock(mouse_mutex_);
            display_scale_ = scale;
        }
        cv::resizeWindow(window_name_, display.cols, display.rows);
        cv::imshow(window_name_, display);

        int key = cv::waitKey(1);
        if (key == 'r' || key == 'R') {
            view_mode_ = ViewMode::FULL;
            follow_robot_ = false;
            view_col_min_ = view_col_max_ = view_row_min_ = view_row_max_ = -1;
            RCLCPP_INFO(get_logger(), "Reset to full view");
        } else if (key == 'f' || key == 'F') {
            follow_robot_ = !follow_robot_;
            view_mode_ = follow_robot_ ? ViewMode::FOLLOW : view_mode_;
            RCLCPP_INFO(get_logger(), "Follow robot: %s", follow_robot_ ? "ON" : "OFF");
        } else if (key == 'm' || key == 'M') {
            use_model_overlay_ = !use_model_overlay_;
            setRemoteParam("use_model_overlay", use_model_overlay_);
            RCLCPP_INFO(get_logger(), "Model overlay: %s", use_model_overlay_ ? "ON" : "OFF");
        } else if (key == 'w' || key == 'W') {
            corridor_width_cm_ = std::min(corridor_width_cm_ + 1, 100);
            setRemoteParam("astar_corridor_half_width_live", corridor_width_cm_ * 0.01);
            RCLCPP_INFO(get_logger(), "Corridor half-width: %.2f m", corridor_width_cm_ * 0.01);
        } else if (key == 's' || key == 'S') {
            corridor_width_cm_ = std::max(corridor_width_cm_ - 1, 0);
            setRemoteParam("astar_corridor_half_width_live", corridor_width_cm_ * 0.01);
            RCLCPP_INFO(get_logger(), "Corridor half-width: %.2f m", corridor_width_cm_ * 0.01);
        } else if ((key == 'o' || key == 'O') && door_toggle_enabled_ && pub_door_toggle_) {
            door_map_index_ = (door_map_index_ == 1) ? 2 : 1;
            std_msgs::msg::String msg;
            msg.data = std::to_string(door_map_index_);
            pub_door_toggle_->publish(msg);
            RCLCPP_INFO(get_logger(), "Door toggle -> map %d", door_map_index_);
        }
    }

    std::mutex mutex_;
    OccData occ_;
    std::vector<cv::Point2d> path_poses_;
    std::vector<cv::Point2d> waypoint_poses_;
    std::optional<cv::Point2d> robot_pose_;
    std::optional<cv::Point2d> goal_pose_;
    std::string planner_status_text_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_occ_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_waypoints_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_planner_status_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_;
    rclcpp::TimerBase::SharedPtr timer_;

    double rate_hz_ = 10.0;
    int view_col_min_ = -1;
    int view_col_max_ = -1;
    int view_row_min_ = -1;
    int view_row_max_ = -1;
    bool show_energy_map_ = true;
    bool show_agent_centered_roi_ = false;
    bool show_robot_marker_ = true;
    bool show_goal_marker_ = true;
    double follow_radius_m_ = 5.0;
    const std::string window_name_ = "Occupancy grid + waypoints (map frame)";

    std::mutex mouse_mutex_;
    int mouse_x_ = -1;
    int mouse_y_ = -1;
    bool has_mouse_ = false;
    int click_x_ = -1;
    int click_y_ = -1;
    bool has_click_ = false;

    // Click-drag zoom and follow mode
    ViewMode view_mode_ = ViewMode::FULL;
    bool follow_robot_ = false;
    bool is_dragging_ = false;
    int drag_start_x_ = 0;
    int drag_start_y_ = 0;
    int drag_end_x_ = 0;
    int drag_end_y_ = 0;
    int last_col_min_ = 0;
    int last_row_min_ = 0;
    int last_occ_panel_width_ = 0;
    int last_occ_panel_height_ = 0;
    double display_scale_ = 1.0;

    // Live controls
    rclcpp::Client<rcl_interfaces::srv::SetParameters>::SharedPtr param_client_;
    bool use_model_overlay_ = true;
    int corridor_width_cm_ = 20;  // in centimeters, displayed as meters (20 = 0.20m)

    // Door toggle
    bool door_toggle_enabled_ = false;
    int door_map_index_ = 1;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_door_toggle_;
};

int main(int argc, char** argv) {
    // Force line-buffered stdout/stderr to prevent display issues when output is piped
    // This fixes the "| grep err" workaround symptom where piping affects OpenCV GTK init
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<WaypointsOccVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>
#include <optional>

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
        rate_hz_ = declare_parameter<double>("rate", 10.0);
        view_col_min_ = declare_parameter<int>("view_col_min", -1);
        view_col_max_ = declare_parameter<int>("view_col_max", -1);
        view_row_min_ = declare_parameter<int>("view_row_min", -1);
        view_row_max_ = declare_parameter<int>("view_row_max", -1);
        show_energy_map_ = declare_parameter<bool>("show_energy_map", true);
        sub_occ_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            occ_topic, 10, [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                onOcc(msg);
            });
        sub_path_ = create_subscription<nav_msgs::msg::Path>(
            path_topic, 10, [this](const nav_msgs::msg::Path::SharedPtr msg) {
                onPath(msg, path_poses_);
            });
        sub_waypoints_ = create_subscription<nav_msgs::msg::Path>(
            waypoints_topic, 10, [this](const nav_msgs::msg::Path::SharedPtr msg) {
                onPath(msg, waypoint_poses_);
            });
        if (!pose_topic.empty()) {
            sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                pose_topic, 10,
                [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                    onPose(msg);
                });
        }
        if (!goal_topic.empty()) {
            sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
                goal_topic, 10,
                [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                    onGoal(msg);
                });
            pub_goal_ = create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic, 10);
        }

        timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate_hz_)),
            [this]() { render(); });

        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, 900, 900);
        cv::setMouseCallback(window_name_, &WaypointsOccVisualizer::onMouse, this);
        RCLCPP_INFO(get_logger(), "Left-click on the visualization window to set a goal!");
    }

private:
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

    bool worldToPixel(const OccData& occ, const cv::Point2d& w, cv::Point& p) const {
        if (occ.width <= 0 || occ.height <= 0 || occ.resolution <= 0.0) return false;
        double col = (w.x - occ.origin_x) / occ.resolution;
        double row = (occ.origin_y + occ.height * occ.resolution - w.y) / occ.resolution;
        int c = static_cast<int>(std::round(col));
        int r = static_cast<int>(std::round(row));
        if (c < 0 || c >= occ.width || r < 0 || r >= occ.height) return false;
        p = cv::Point(c, r);
        return true;
    }

    cv::Point2d pixelToWorld(const OccData& occ, int col, int row) const {
        double x = occ.origin_x + col * occ.resolution;
        double y = occ.origin_y + (occ.height - 1 - row) * occ.resolution;
        return cv::Point2d(x, y);
    }

    static void onMouse(int event, int x, int y, int, void* userdata) {
        auto* self = static_cast<WaypointsOccVisualizer*>(userdata);
        
        if (event == cv::EVENT_LBUTTONDOWN) {
            std::lock_guard<std::mutex> lock(self->mouse_mutex_);
            self->click_x_ = x;
            self->click_y_ = y;
            self->has_click_ = true;
            return;
        }
        
        if (event == cv::EVENT_MOUSEMOVE) {
            std::lock_guard<std::mutex> lock(self->mouse_mutex_);
            self->mouse_x_ = x;
            self->mouse_y_ = y;
            self->has_mouse_ = true;
        }
    }

    void drawLabel(cv::Mat& img, const std::string& text) const {
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::Rect bg(5, 5, text_size.width + 8, text_size.height + baseline + 8);
        cv::rectangle(img, bg, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img, text, cv::Point(9, 5 + text_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    }

    void render() {
        OccData occ;
        std::vector<cv::Point2d> path;
        std::vector<cv::Point2d> waypoints;
        std::optional<cv::Point2d> robot;
        std::optional<cv::Point2d> goal;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            occ = occ_;
            path = path_poses_;
            waypoints = waypoint_poses_;
            robot = robot_pose_;
            goal = goal_pose_;
        }

        if (occ.width == 0 || occ.height == 0 || occ.data.empty()) {
            return;
        }

        cv::Mat img(occ.height, occ.width, CV_8UC1);
        for (int r = 0; r < occ.height; ++r) {
            for (int c = 0; c < occ.width; ++c) {
                int idx = r * occ.width + c;
                int8_t val = occ.data[idx];
                if (val < 0) {
                    img.at<uchar>(r, c) = 127;
                } else {
                    int v = 100 - val;
                    v = std::max(0, std::min(100, v));
                    img.at<uchar>(r, c) = static_cast<uchar>(v * 255 / 100);
                }
            }
        }

        cv::Mat vis;
        cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);

        // Optional: build an "energy" map that matches occupancy_grid_planning_energy_opt.py:
        // distance transform of free space visualized with a viridis-style colormap.
        cv::Mat energy_color;
        if (show_energy_map_) {
            cv::Mat binary_inv(occ.height, occ.width, CV_8UC1);
            for (int r = 0; r < occ.height; ++r) {
                for (int c = 0; c < occ.width; ++c) {
                    int idx = r * occ.width + c;
                    int8_t val = occ.data[idx];
                    // OccupancyGrid: 0 = free, >0 (e.g., 100 or -1) treated as obstacle.
                    bool free_cell = (val == 0);
                    binary_inv.at<uchar>(r, c) = free_cell ? 255 : 0;
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
            cv::applyColorMap(dist_norm_u8, energy_color, cv::COLORMAP_VIRIDIS);
        }

        int col_min = 0;
        int col_max = occ.width - 1;
        int row_min = 0;
        int row_max = occ.height - 1;
        if (view_col_min_ >= 0) col_min = view_col_min_;
        if (view_col_max_ >= 0) col_max = view_col_max_;
        if (view_row_min_ >= 0) row_min = view_row_min_;
        if (view_row_max_ >= 0) row_max = view_row_max_;

        col_min = std::max(0, std::min(col_min, occ.width - 1));
        col_max = std::max(0, std::min(col_max, occ.width - 1));
        row_min = std::max(0, std::min(row_min, occ.height - 1));
        row_max = std::max(0, std::min(row_max, occ.height - 1));
        if (col_max < col_min) std::swap(col_min, col_max);
        if (row_max < row_min) std::swap(row_min, row_max);

        cv::Rect roi(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1);
        cv::Mat view_occ = vis(roi);
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

        if (robot.has_value()) {
            cv::Point p;
            if (worldToPixel(occ, robot.value(), p)) {
                cv::Point p_local = p - cv::Point(col_min, row_min);
                cv::circle(view_occ, p_local, 6, cv::Scalar(0, 255, 0), -1);
                if (show_energy_map_ && !view_energy.empty()) {
                    cv::circle(view_energy, p_local, 6, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        if (goal.has_value()) {
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
        if (mouse_x >= 0 && mouse_y >= 0 &&
            mouse_x < occ_width && mouse_y < occ_height) {
            int col = col_min + mouse_x;
            int row = row_min + mouse_y;
            double x = occ.origin_x + col * occ.resolution;
            double y = occ.origin_y + (occ.height - 1 - row) * occ.resolution;
            drawLabel(view_occ, "col,row: " + std::to_string(col) + ", " + std::to_string(row) +
                              "  x,y: " + cv::format("%.2f, %.2f", x, y));
        } else {
            drawLabel(view_occ, "col,row: out of bounds");
        }

        // Build single combined view: occupancy on the left, energy map on the right
        cv::Mat combined;
        if (show_energy_map_ && !view_energy.empty()) {
            cv::hconcat(view_occ, view_energy, combined);
        } else {
            combined = view_occ;
        }

        cv::resizeWindow(window_name_, combined.cols, combined.rows);
        cv::imshow(window_name_, combined);
        cv::waitKey(1);
    }

    std::mutex mutex_;
    OccData occ_;
    std::vector<cv::Point2d> path_poses_;
    std::vector<cv::Point2d> waypoint_poses_;
    std::optional<cv::Point2d> robot_pose_;
    std::optional<cv::Point2d> goal_pose_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_occ_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_waypoints_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_;
    rclcpp::TimerBase::SharedPtr timer_;

    double rate_hz_ = 10.0;
    int view_col_min_ = -1;
    int view_col_max_ = -1;
    int view_row_min_ = -1;
    int view_row_max_ = -1;
    bool show_energy_map_ = true;
    const std::string window_name_ = "Occupancy grid + waypoints (map frame)";

    std::mutex mouse_mutex_;
    int mouse_x_ = -1;
    int mouse_y_ = -1;
    bool has_mouse_ = false;
    int click_x_ = -1;
    int click_y_ = -1;
    bool has_click_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WaypointsOccVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

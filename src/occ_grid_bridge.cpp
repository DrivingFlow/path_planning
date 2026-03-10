#include "occ_grid_bridge.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace {

bool loadPcdFile(const std::string& path, std::vector<std::array<double, 3>>& out_points) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    bool binary = false;
    int num_points = 0;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line.find("POINTS") != std::string::npos) {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok >> num_points;
        }
        if (line.find("DATA") != std::string::npos) {
            binary = (line.find("binary") != std::string::npos);
            break;
        }
    }
    if (num_points <= 0) return false;
    out_points.reserve(static_cast<size_t>(num_points));

    if (binary) {
        float buf[6];
        for (int i = 0; i < num_points && f.read(reinterpret_cast<char*>(buf), sizeof(buf)); ++i) {
            out_points.push_back({{static_cast<double>(buf[0]), static_cast<double>(buf[1]), static_cast<double>(buf[2])}});
        }
    } else {
        double x, y, z;
        while (f >> x >> y >> z)
            out_points.push_back({{x, y, z}});
    }
    return !out_points.empty();
}

} // namespace

bool OccGridBridge::loadMapPcd(const std::string& pcd_path, double resolution) {
    std::vector<std::array<double, 3>> map_points;
    if (!loadPcdFile(pcd_path, map_points))
        return false;

    double x_min = 1e30, x_max = -1e30, y_min = 1e30, y_max = -1e30;
    for (const auto& pt : map_points) {
        x_min = std::min(x_min, pt[0]);
        x_max = std::max(x_max, pt[0]);
        y_min = std::min(y_min, pt[1]);
        y_max = std::max(y_max, pt[1]);
    }

    x_min_ = x_min;
    x_max_ = x_max;
    y_min_ = y_min;
    y_max_ = y_max;
    res_ = resolution;
    computeGridSize();
    initialized_ = true;
    return true;
}

void OccGridBridge::setBounds(double x_min, double x_max, double y_min, double y_max, double resolution) {
    x_min_ = x_min;
    x_max_ = x_max;
    y_min_ = y_min;
    y_max_ = y_max;
    res_ = resolution;
    computeGridSize();
    initialized_ = true;
}

void OccGridBridge::computeGridSize() {
    w_ = static_cast<int>((x_max_ - x_min_) / res_) + 1;
    h_ = static_cast<int>((y_max_ - y_min_) / res_) + 1;
    w_ = std::max(1, w_);
    h_ = std::max(1, h_);
}

bool OccGridBridge::loadEditedMapPng(const std::string& png_path) {
    cv::Mat img = cv::imread(png_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) return false;
    // File: white=free (255), black=obstacle (0). We store 0=free, 255=obstacle for RRT.
    static_map_ = 255 - img;
    if (w_ > 0 && h_ > 0 && (img.cols != w_ || img.rows != h_)) {
        // If PNG size differs from bounds-derived grid, adjust bounds to match PNG size.
        // This avoids hard failure when grid sizing differs by rounding or off-by-one.
        w_ = img.cols;
        h_ = img.rows;
        x_max_ = x_min_ + (w_ - 1) * res_;
        y_max_ = y_min_ + (h_ - 1) * res_;
    }
    return true;
}

void OccGridBridge::worldToGrid(double x, double y, int& col, int& row) const {
    col = static_cast<int>((x - x_min_) / res_);
    row = h_ - 1 - static_cast<int>((y - y_min_) / res_);
}

void OccGridBridge::gridToWorld(int col, int row, double& x, double& y) const {
    x = x_min_ + col * res_;
    y = y_min_ + (h_ - 1 - row) * res_;
}

cv::Mat OccGridBridge::pointcloudToOccupancyGrid(
    const std::vector<std::array<float, 3>>& points_xyz,
    double z_min, double z_max) const {
    int n = 0;
    for (const auto& pt : points_xyz)
        if (pt[2] >= z_min && pt[2] <= z_max) ++n;
    if (n == 0) return cv::Mat::zeros(h_, w_, CV_8UC1);

    Eigen::Matrix2Xd xy(2, n);
    n = 0;
    for (const auto& pt : points_xyz) {
        if (pt[2] >= z_min && pt[2] <= z_max) {
            xy(0, n) = pt[0];
            xy(1, n) = pt[1];
            ++n;
        }
    }

    cv::Mat grid = cv::Mat::zeros(h_, w_, CV_8UC1);
    const double inv_res = 1.0 / res_;
    const double y_off = h_ - 1 + y_min_ * inv_res;
    for (int i = 0; i < n; ++i) {
        int col = static_cast<int>((xy(0, i) - x_min_) * inv_res);
        int row = static_cast<int>(y_off - xy(1, i) * inv_res);
        if (col >= 0 && col < w_ && row >= 0 && row < h_)
            grid.ptr<uchar>(row)[col] = 255;
    }
    return grid;
}

cv::Mat OccGridBridge::mergeWithStaticMap(const cv::Mat& live_occupancy) const {
    if (static_map_.empty())
        return live_occupancy.clone();
    cv::Mat combined;
    cv::max(static_map_, live_occupancy, combined);
    return combined;
}

cv::Mat OccGridBridge::pointcloudToEgoOccupancyGrid201(
    const std::vector<std::array<float, 3>>& points_xyz,
    double robot_x, double robot_y, double robot_yaw,
    double z_min, double z_max) {
    const int sz = EGO_GRID_SIZE;
    const double res = egoGridResolution();
    const double ox = egoGridOriginX();
    const double oy = egoGridOriginY();
    const double r2_max = EGO_RADIUS_M * EGO_RADIUS_M;

    // Filter by z and fill 2xN delta for batch rotation
    int n = 0;
    for (const auto& pt : points_xyz)
        if (pt[2] >= z_min && pt[2] <= z_max) ++n;
    if (n == 0) return cv::Mat::zeros(sz, sz, CV_8UC1);

    Eigen::Matrix2Xd delta(2, n);
    n = 0;
    for (const auto& pt : points_xyz) {
        if (pt[2] >= z_min && pt[2] <= z_max) {
            delta(0, n) = pt[0] - robot_x;
            delta(1, n) = pt[1] - robot_y;
            ++n;
        }
    }

    Eigen::Matrix2d R;
    R << std::cos(robot_yaw), std::sin(robot_yaw),
         -std::sin(robot_yaw), std::cos(robot_yaw);
    Eigen::Matrix2Xd ego = R * delta;  // Single batch multiply: 2x2 * 2xN = 2xN

    cv::Mat grid = cv::Mat::zeros(sz, sz, CV_8UC1);
    for (int i = 0; i < n; ++i) {
        double ex = ego(0, i), ey = ego(1, i);
        if (ex * ex + ey * ey > r2_max) continue;
        int col = static_cast<int>((ex - ox) / res);
        int row = static_cast<int>((oy - ey) / res);
        if (col >= 0 && col < sz && row >= 0 && row < sz)
            grid.ptr<uchar>(row)[col] = 255;
    }
    return grid;
}

void OccGridBridge::pasteEgoGridIntoMap(const cv::Mat& ego_grid,
                                        double anchor_x, double anchor_y, double anchor_yaw,
                                        cv::Mat& map_out) const {
    if (ego_grid.rows != EGO_GRID_SIZE || ego_grid.cols != EGO_GRID_SIZE || map_out.empty())
        return;
    const double res_ego = egoGridResolution();
    const double ox_ego = egoGridOriginX();
    const double oy_ego = egoGridOriginY();
    const double r2_max = EGO_RADIUS_M * EGO_RADIUS_M;

    Eigen::Matrix2d R;
    R << std::cos(anchor_yaw), -std::sin(anchor_yaw),
         std::sin(anchor_yaw), std::cos(anchor_yaw);
    Eigen::Vector2d anchor(anchor_x, anchor_y);

    for (int r = 0; r < EGO_GRID_SIZE; ++r) {
        const uchar* row_ptr = ego_grid.ptr<uchar>(r);
        for (int c = 0; c < EGO_GRID_SIZE; ++c) {
            if (row_ptr[c] == 0) continue;
            Eigen::Vector2d ego(ox_ego + c * res_ego, oy_ego - r * res_ego);
            if (ego.squaredNorm() > r2_max) continue;
            Eigen::Vector2d map_pt = R * ego + anchor;
            int col, row;
            worldToGrid(map_pt.x(), map_pt.y(), col, row);
            if (col >= 0 && col < w_ && row >= 0 && row < h_) {
                uchar& cell = map_out.ptr<uchar>(row)[col];
                if (255 > cell) cell = 255;
            }
        }
    }
}

void OccGridBridge::zeroEgoFootprintInMap(double anchor_x, double anchor_y, double /*anchor_yaw*/,
                                          cv::Mat& map_out) const {
    if (map_out.empty() || map_out.cols != w_ || map_out.rows != h_)
        return;

    // Iterate over every map pixel whose world position falls within EGO_RADIUS_M of the anchor.
    // This is rotation-independent and guarantees no map cell inside the circle is missed
    // (the old approach of iterating over ego-grid pixels left sub-pixel gaps after rotation).
    const double r2_max = EGO_RADIUS_M * EGO_RADIUS_M;

    // Compute the map-pixel bounding box of the circle to avoid scanning the whole map.
    int anchor_col, anchor_row;
    worldToGrid(anchor_x, anchor_y, anchor_col, anchor_row);
    int radius_px = static_cast<int>(std::ceil(EGO_RADIUS_M / res_)) + 1;

    int r_min = std::max(0,     anchor_row - radius_px);
    int r_max = std::min(h_ - 1, anchor_row + radius_px);
    int c_min = std::max(0,     anchor_col - radius_px);
    int c_max = std::min(w_ - 1, anchor_col + radius_px);

    Eigen::Vector2d anchor(anchor_x, anchor_y);
    for (int r = r_min; r <= r_max; ++r) {
        uchar* out_row = map_out.ptr<uchar>(r);
        for (int c = c_min; c <= c_max; ++c) {
            double wx, wy;
            gridToWorld(c, r, wx, wy);
            if ((Eigen::Vector2d(wx, wy) - anchor).squaredNorm() <= r2_max)
                out_row[c] = 0;
        }
    }
}

cv::Mat OccGridBridge::staticMapToEgoGrid201(double robot_x, double robot_y, double robot_yaw) const {
    if (static_map_.empty() || static_map_.rows != h_ || static_map_.cols != w_)
        return cv::Mat::zeros(EGO_GRID_SIZE, EGO_GRID_SIZE, CV_8UC1);
    const int sz = EGO_GRID_SIZE;
    const double res_ego = egoGridResolution();
    const double ox_ego = egoGridOriginX();
    const double oy_ego = egoGridOriginY();
    const double r2_max = EGO_RADIUS_M * EGO_RADIUS_M;

    Eigen::Matrix2d R;
    R << std::cos(robot_yaw), -std::sin(robot_yaw),
         std::sin(robot_yaw), std::cos(robot_yaw);
    Eigen::Vector2d robot(robot_x, robot_y);

    cv::Mat grid = cv::Mat::zeros(sz, sz, CV_8UC1);
    for (int r = 0; r < sz; ++r) {
        uchar* out_row = grid.ptr<uchar>(r);
        for (int c = 0; c < sz; ++c) {
            Eigen::Vector2d ego(ox_ego + c * res_ego, oy_ego - r * res_ego);
            if (ego.squaredNorm() > r2_max) continue;
            Eigen::Vector2d map_pt = R * ego + robot;
            int col, row;
            worldToGrid(map_pt.x(), map_pt.y(), col, row);
            if (col >= 0 && col < w_ && row >= 0 && row < h_)
                out_row[c] = static_map_.ptr<uchar>(row)[col];
        }
    }
    return grid;
}

std::vector<std::array<double, 2>> OccGridBridge::pathIndicesToWorld(
    const std::vector<cv::Point2i>& path_indices) const {
    std::vector<std::array<double, 2>> out;
    out.reserve(path_indices.size());
    for (const auto& pt : path_indices) {
        double x, y;
        gridToWorld(pt.x, pt.y, x, y);
        out.push_back({{x, y}});
    }
    return out;
}

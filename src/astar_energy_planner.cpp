#include "astar_energy_planner.hpp"
#include <queue>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

namespace path_planning {

AStarEnergyPlanner::AStarEnergyPlanner(const cv::Mat& grid_img,
                                       int robot_radius_pixels,
                                       int corridor_half_width_pixels)
    : robot_radius_px_(std::max(0, robot_radius_pixels)),
      corridor_half_width_px_(std::max(0, corridor_half_width_pixels)) {
    grid_ = grid_img.clone();
    h_ = grid_.rows;
    w_ = grid_.cols;

    // Binary: 0 = obstacle, 255 = free (for distance to nearest obstacle)
    cv::Mat binary_inv(h_, w_, CV_8UC1);
    for (int r = 0; r < h_; ++r) {
        for (int c = 0; c < w_; ++c) {
            binary_inv.at<uchar>(r, c) = (grid_.at<uchar>(r, c) < 50) ? 255 : 0;
        }
    }

    cv::distanceTransform(binary_inv, dist_, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    dist_.convertTo(dist_, CV_64F);

    const double eps = 1e-6;
    clearance_ = cv::Mat(h_, w_, CV_64F);
    valid_ = cv::Mat(h_, w_, CV_8UC1);
    double max_clear = 0;
    for (int r = 0; r < h_; ++r) {
        for (int c = 0; c < w_; ++c) {
            double d = dist_.at<double>(r, c);
            // Validity uses robot_radius only (obstacle inflation)
            double cl = std::max(d - robot_radius_px_, eps);
            clearance_.at<double>(r, c) = cl;
            valid_.at<uchar>(r, c) = (d > robot_radius_px_) ? 255 : 0;
            if (cl > max_clear) max_clear = cl;
        }
    }
    max_clearance_ = max_clear;
    buildNeighbors();
    computeGradient();
}

void AStarEnergyPlanner::computeGradient() {
    cv::Sobel(clearance_, grad_x_, CV_64F, 1, 0, 3);
    cv::Sobel(clearance_, grad_y_, CV_64F, 0, 1, 3);
}

void AStarEnergyPlanner::buildNeighbors() {
    neighbors_.clear();
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            double cost = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
            neighbors_.push_back({dx, dy, cost});
        }
    }
    const int knight[8][2] = {{2,1},{1,2},{-2,1},{-1,2},{2,-1},{1,-2},{-2,-1},{-1,-2}};
    for (int i = 0; i < 8; ++i) {
        int dx = knight[i][0], dy = knight[i][1];
        double cost = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
        neighbors_.push_back({dx, dy, cost});
    }
}

double AStarEnergyPlanner::heuristic(int ax, int ay, int bx, int by) const {
    return std::hypot(static_cast<double>(bx - ax), static_cast<double>(by - ay));
}

bool AStarEnergyPlanner::edgeCorridorFree(int x0, int y0, int x1, int y1) const {
    // If no corridor check needed, skip entirely
    if (corridor_half_width_px_ <= 0) return true;

    double dx = static_cast<double>(x1 - x0);
    double dy = static_cast<double>(y1 - y0);
    double seg_len = std::hypot(dx, dy);
    int n_samples = std::max(1, static_cast<int>(std::ceil(seg_len / 0.5)));

    for (int i = 0; i <= n_samples; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n_samples);
        int x = static_cast<int>(std::round(static_cast<double>(x0) + t * dx));
        int y = static_cast<int>(std::round(static_cast<double>(y0) + t * dy));
        if (x < 0 || x >= w_ || y < 0 || y >= h_) return false;
        // Edge checking uses corridor_half_width only (independent of robot_radius)
        if (dist_.at<double>(y, x) <= corridor_half_width_px_) return false;
    }
    return true;
}

std::vector<cv::Point2i> AStarEnergyPlanner::plan(const cv::Point2f& start, const cv::Point2f& goal,
                                                  double beta_valley,
                                                  double smooth_alpha,
                                                  double smooth_beta,
                                                  int smooth_n_iter,
                                                  double resample_ds) {
    int sx = static_cast<int>(std::round(start.x));
    int sy = static_cast<int>(std::round(start.y));
    int gx = static_cast<int>(std::round(goal.x));
    int gy = static_cast<int>(std::round(goal.y));

    if (sx < 0 || sx >= w_ || sy < 0 || sy >= h_) return {};
    if (gx < 0 || gx >= w_ || gy < 0 || gy >= h_) return {};
    if (valid_.at<uchar>(sy, sx) == 0) return {};
    if (valid_.at<uchar>(gy, gx) == 0) return {};

    cv::Mat g_score = cv::Mat::ones(h_, w_, CV_64F) * std::numeric_limits<double>::infinity();
    cv::Mat parent_x = cv::Mat::ones(h_, w_, CV_32SC1) * -1;
    cv::Mat parent_y = cv::Mat::ones(h_, w_, CV_32SC1) * -1;
    cv::Mat visited = cv::Mat::zeros(h_, w_, CV_8UC1);

    struct Entry {
        double f;
        int x, y;
        bool operator>(const Entry& o) const { return f > o.f; }
    };
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    g_score.at<double>(sy, sx) = 0;
    pq.push({heuristic(sx, sy, gx, gy), sx, sy});

    while (!pq.empty()) {
        Entry e = pq.top();
        pq.pop();
        int cx = e.x, cy = e.y;

        if (visited.at<uchar>(cy, cx)) continue;
        visited.at<uchar>(cy, cx) = 255;

        if (cx == gx && cy == gy) break;

        for (const auto& n : neighbors_) {
            int nx = cx + n.dx;
            int ny = cy + n.dy;
            if (nx < 0 || nx >= w_ || ny < 0 || ny >= h_) continue;
            if (valid_.at<uchar>(ny, nx) == 0) continue;
            if (!edgeCorridorFree(cx, cy, nx, ny)) continue;

            double cl = clearance_.at<double>(ny, nx);
            double norm_clear = (max_clearance_ > 0) ? (cl / max_clearance_) : 1.0;
            double energy_factor = std::exp(-beta_valley * norm_clear);
            double cost = n.cost * energy_factor;

            double tentative_g = g_score.at<double>(cy, cx) + cost;
            if (tentative_g < g_score.at<double>(ny, nx)) {
                g_score.at<double>(ny, nx) = tentative_g;
                double f = tentative_g + heuristic(nx, ny, gx, gy);
                pq.push({f, nx, ny});
                parent_x.at<int>(ny, nx) = cx;
                parent_y.at<int>(ny, nx) = cy;
            }
        }
    }

    if (g_score.at<double>(gy, gx) == std::numeric_limits<double>::infinity())
        return {};

    std::vector<cv::Point2i> path;
    int cur_x = gx, cur_y = gy;
    while (cur_x >= 0 && cur_y >= 0) {
        path.push_back(cv::Point2i(cur_x, cur_y));
        if (cur_x == sx && cur_y == sy) break;
        int px = parent_x.at<int>(cur_y, cur_x);
        int py = parent_y.at<int>(cur_y, cur_x);
        cur_x = px;
        cur_y = py;
    }
    std::reverse(path.begin(), path.end());

    if (path.size() < 2) return path;

    // Convert to float for smoothing and resampling
    std::vector<cv::Point2f> path_f(path.size());
    for (size_t i = 0; i < path.size(); ++i)
        path_f[i] = cv::Point2f(static_cast<float>(path[i].x), static_cast<float>(path[i].y));

    auto polylineCorridorFree = [this](const std::vector<cv::Point2f>& polyline) {
        if (polyline.size() < 2) return true;
        for (size_t i = 0; i + 1 < polyline.size(); ++i) {
            int x0 = static_cast<int>(std::round(polyline[i].x));
            int y0 = static_cast<int>(std::round(polyline[i].y));
            int x1 = static_cast<int>(std::round(polyline[i + 1].x));
            int y1 = static_cast<int>(std::round(polyline[i + 1].y));
            if (!edgeCorridorFree(x0, y0, x1, y1)) return false;
        }
        return true;
    };

    std::vector<cv::Point2f> smoothed = smoothPath(path_f, smooth_alpha, smooth_beta, smooth_n_iter);
    smoothed = resamplePath(smoothed, resample_ds);
    if (polylineCorridorFree(smoothed)) {
        path_f = std::move(smoothed);
    }

    std::vector<cv::Point2i> result;
    result.reserve(path_f.size());
    for (const auto& p : path_f)
        result.push_back(cv::Point2i(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))));
    return result;
}

std::vector<cv::Point2f> AStarEnergyPlanner::smoothPath(std::vector<cv::Point2f> path,
                                                        double alpha, double beta, int n_iter) const {
    if (path.size() < 3 || n_iter <= 0) return path;

    std::vector<Eigen::Vector2d> pts(path.size());
    for (size_t i = 0; i < path.size(); ++i)
        pts[i] = Eigen::Vector2d(static_cast<double>(path[i].x), static_cast<double>(path[i].y));

    for (int it = 0; it < n_iter; ++it) {
        std::vector<Eigen::Vector2d> new_pts = pts;
        for (size_t i = 1; i + 1 < pts.size(); ++i) {
            const Eigen::Vector2d& prev_pt = pts[i - 1];
            const Eigen::Vector2d& next_pt = pts[i + 1];
            const Eigen::Vector2d& curr = pts[i];

            Eigen::Vector2d smooth_force = alpha * (prev_pt + next_pt - 2.0 * curr);

            int xi = static_cast<int>(curr.x()), yi = static_cast<int>(curr.y());
            xi = std::max(0, std::min(w_ - 1, xi));
            yi = std::max(0, std::min(h_ - 1, yi));
            double gx = grad_x_.at<double>(yi, xi);
            double gy = grad_y_.at<double>(yi, xi);
            Eigen::Vector2d repel_force(beta * gx, beta * gy);

            Eigen::Vector2d new_pt = curr + smooth_force + repel_force;
            int nx = static_cast<int>(new_pt.x()), ny = static_cast<int>(new_pt.y());
            if (nx >= 0 && nx < w_ && ny >= 0 && ny < h_ && valid_.at<uchar>(ny, nx) != 0)
                new_pts[i] = new_pt;
        }
        pts = new_pts;
    }

    std::vector<cv::Point2f> result(path.size());
    for (size_t i = 0; i < pts.size(); ++i)
        result[i] = cv::Point2f(static_cast<float>(pts[i].x()), static_cast<float>(pts[i].y()));
    return result;
}

std::vector<cv::Point2f> AStarEnergyPlanner::resamplePath(const std::vector<cv::Point2f>& path, double ds) const {
    if (path.size() < 2 || ds <= 0) return path;

    std::vector<Eigen::Vector2d> pts(path.size());
    for (size_t i = 0; i < path.size(); ++i)
        pts[i] = Eigen::Vector2d(static_cast<double>(path[i].x), static_cast<double>(path[i].y));

    std::vector<double> seg_lengths(path.size() - 1);
    std::vector<double> s(path.size());
    s[0] = 0;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        seg_lengths[i] = (pts[i + 1] - pts[i]).norm();
        s[i + 1] = s[i] + seg_lengths[i];
    }
    double total_length = s.back();
    if (total_length < 1e-9) return path;
    if (total_length < ds) return {path[0], path.back()};

    std::vector<cv::Point2f> new_pts;
    size_t seg_idx = 0;
    for (double si = 0; ; si += ds) {
        if (si > total_length) si = total_length;
        while (seg_idx + 1 < seg_lengths.size() && s[seg_idx + 1] < si)
            seg_idx++;
        double seg_len = std::max(seg_lengths[seg_idx], 1e-9);
        double t = (si - s[seg_idx]) / seg_len;
        Eigen::Vector2d p = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx]);
        new_pts.push_back(cv::Point2f(static_cast<float>(p.x()), static_cast<float>(p.y())));
        if (si >= total_length) break;
    }
    return new_pts;
}

} // namespace path_planning

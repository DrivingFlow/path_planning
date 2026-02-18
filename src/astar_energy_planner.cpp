#include "astar_energy_planner.hpp"
#include <queue>
#include <cmath>
#include <limits>

namespace path_planning {

AStarEnergyPlanner::AStarEnergyPlanner(const cv::Mat& grid_img, int robot_radius_pixels)
    : robot_radius_px_(robot_radius_pixels) {
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

    path_f = smoothPath(path_f, smooth_alpha, smooth_beta, smooth_n_iter);
    path_f = resamplePath(path_f, resample_ds);

    std::vector<cv::Point2i> result;
    result.reserve(path_f.size());
    for (const auto& p : path_f)
        result.push_back(cv::Point2i(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))));
    return result;
}

std::vector<cv::Point2f> AStarEnergyPlanner::smoothPath(std::vector<cv::Point2f> path,
                                                        double alpha, double beta, int n_iter) const {
    if (path.size() < 3 || n_iter <= 0) return path;

    for (int it = 0; it < n_iter; ++it) {
        std::vector<cv::Point2f> new_path = path;
        for (size_t i = 1; i + 1 < path.size(); ++i) {
            const cv::Point2f& prev_pt = path[i - 1];
            const cv::Point2f& next_pt = path[i + 1];
            const cv::Point2f& curr = path[i];

            cv::Point2f smooth_force(alpha * (prev_pt.x + next_pt.x - 2.f * curr.x),
                                    alpha * (prev_pt.y + next_pt.y - 2.f * curr.y));

            int xi = static_cast<int>(curr.x), yi = static_cast<int>(curr.y);
            xi = std::max(0, std::min(w_ - 1, xi));
            yi = std::max(0, std::min(h_ - 1, yi));
            double gx = grad_x_.at<double>(yi, xi);
            double gy = grad_y_.at<double>(yi, xi);
            cv::Point2f repel_force(static_cast<float>(beta * gx), static_cast<float>(beta * gy));

            cv::Point2f new_pt = curr + smooth_force + repel_force;
            int nx = static_cast<int>(new_pt.x), ny = static_cast<int>(new_pt.y);
            if (nx >= 0 && nx < w_ && ny >= 0 && ny < h_ && valid_.at<uchar>(ny, nx) != 0)
                new_path[i] = new_pt;
        }
        path = new_path;
    }
    return path;
}

std::vector<cv::Point2f> AStarEnergyPlanner::resamplePath(const std::vector<cv::Point2f>& path, double ds) const {
    if (path.size() < 2 || ds <= 0) return path;

    std::vector<double> seg_lengths(path.size() - 1);
    std::vector<double> s(path.size());
    s[0] = 0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        double dx = path[i + 1].x - path[i].x;
        double dy = path[i + 1].y - path[i].y;
        seg_lengths[i] = std::sqrt(dx * dx + dy * dy);
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
        float x = path[seg_idx].x + static_cast<float>(t) * (path[seg_idx + 1].x - path[seg_idx].x);
        float y = path[seg_idx].y + static_cast<float>(t) * (path[seg_idx + 1].y - path[seg_idx].y);
        new_pts.push_back(cv::Point2f(x, y));
        if (si >= total_length) break;
    }
    return new_pts;
}

} // namespace path_planning

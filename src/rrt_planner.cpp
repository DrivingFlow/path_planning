#include "rrt_planner.hpp"

namespace path_planning {

RRTPlanner::RRTPlanner(const cv::Mat& grid_img,
                       double step_size,
                       double goal_sample_rate,
                       int robot_radius,
                       double rewire_gamma,
                       int sample_col_min,
                       int sample_col_max,
                       int sample_row_min,
                       int sample_row_max)
    : grid_(grid_img.clone()),
      step_size_(step_size),
      goal_sample_rate_(goal_sample_rate),
      rewire_gamma_(rewire_gamma),
      rng_(std::random_device{}()),
      uniform_dist_(0.0, 1.0) {
    h_ = grid_.rows;
    w_ = grid_.cols;
    sample_col_min_ = sample_col_min;
    sample_col_max_ = sample_col_max;
    sample_row_min_ = sample_row_min;
    sample_row_max_ = sample_row_max;
    cv::Mat binary = grid_ >= 50;
    int kernel_size = 2 * robot_radius + 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::dilate(binary, occ_, kernel);
}

double RRTPlanner::sqdist(const cv::Point2f& a, const cv::Point2f& b) const {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

double RRTPlanner::dist(const cv::Point2f& a, const cv::Point2f& b) const {
    return std::sqrt(sqdist(a, b));
}

int RRTPlanner::findNearest(const cv::Point2f& query, const std::vector<Node>& nodes) const {
    int best_idx = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < nodes.size(); ++i) {
        double d = sqdist(query, nodes[i].pos);
        if (d < best_dist) { best_dist = d; best_idx = static_cast<int>(i); }
    }
    return best_idx;
}

std::vector<int> RRTPlanner::findNear(const cv::Point2f& query, double radius,
                                      const std::vector<Node>& nodes) const {
    std::vector<int> near_idxs;
    double radius_sq = radius * radius;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (sqdist(query, nodes[i].pos) <= radius_sq)
            near_idxs.push_back(static_cast<int>(i));
    }
    return near_idxs;
}

cv::Mat RRTPlanner::binaryDilation(const cv::Mat& binary, int kernel_size) const {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat dilated;
    cv::dilate(binary, dilated, kernel);
    return dilated;
}

bool RRTPlanner::collisionFree(const cv::Point2f& a, const cv::Point2f& b) const {
    double length = dist(a, b);
    int n = static_cast<int>(length / 3.0) + 1;
    for (int i = 0; i <= n; ++i) {
        double t = static_cast<double>(i) / n;
        int x = static_cast<int>(a.x + t * (b.x - a.x));
        int y = static_cast<int>(a.y + t * (b.y - a.y));
        if (x < 0 || y < 0 || x >= w_ || y >= h_) return false;
        if (occ_.at<uchar>(y, x) != 0) return false;
    }
    return true;
}

cv::Point2f RRTPlanner::uniformSample() const {
    double x_lo = 0.0, x_hi = static_cast<double>(w_);
    double y_lo = 0.0, y_hi = static_cast<double>(h_);
    if (sample_col_min_ >= 0 && sample_col_max_ >= 0) {
        x_lo = static_cast<double>(std::max(0, sample_col_min_));
        x_hi = static_cast<double>(std::min(w_ - 1, sample_col_max_)) + 1.0;
    }
    if (sample_row_min_ >= 0 && sample_row_max_ >= 0) {
        y_lo = static_cast<double>(std::max(0, sample_row_min_));
        y_hi = static_cast<double>(std::min(h_ - 1, sample_row_max_)) + 1.0;
    }
    std::uniform_real_distribution<double> x_dist(x_lo, x_hi);
    std::uniform_real_distribution<double> y_dist(y_lo, y_hi);
    return cv::Point2f(static_cast<float>(x_dist(rng_)), static_cast<float>(y_dist(rng_)));
}

bool RRTPlanner::informedFilter(const cv::Point2f& x, const cv::Point2f& start,
                                const cv::Point2f& goal, double max_sum_dist) const {
    if (max_sum_dist < 0) return true;
    return dist(x, start) + dist(x, goal) <= max_sum_dist;
}

cv::Point2f RRTPlanner::steer(const cv::Point2f& a, const cv::Point2f& b) const {
    cv::Point2f v = b - a;
    double d = cv::norm(v);
    if (d <= step_size_) return b;
    return a + (v / static_cast<float>(d)) * static_cast<float>(step_size_);
}

std::vector<cv::Point2f> RRTPlanner::resamplePath(const std::vector<cv::Point2f>& path, double ds) const {
    if (path.size() < 2) return path;
    std::vector<double> seg_lengths, s;
    s.push_back(0.0);
    for (size_t i = 0; i < path.size() - 1; ++i) {
        double seg_len = dist(path[i], path[i + 1]);
        seg_lengths.push_back(seg_len);
        s.push_back(s.back() + seg_len);
    }
    double total_length = s.back();
    if (total_length < ds) return {path[0], path.back()};

    std::vector<double> s_new;
    for (double si = 0.0; si < total_length; si += ds) s_new.push_back(si);
    s_new.push_back(total_length);

    std::vector<cv::Point2f> new_pts;
    size_t seg_idx = 0;
    for (double si : s_new) {
        while (seg_idx < seg_lengths.size() - 1 && s[seg_idx + 1] < si) seg_idx++;
        double seg_len = std::max(seg_lengths[seg_idx], 1e-9);
        double t = (si - s[seg_idx]) / seg_len;
        new_pts.push_back(path[seg_idx] + static_cast<float>(t) * (path[seg_idx + 1] - path[seg_idx]));
    }
    return new_pts;
}

std::vector<cv::Point2i> RRTPlanner::plan(const cv::Point2f& start, const cv::Point2f& goal,
                                          int n_iter, bool debug, bool plot) {
    std::vector<Node> nodes;
    nodes.push_back(Node(start, -1, 0.0));
    int best_goal_idx = -1;
    double best_cost = -1.0;
    double step2 = step_size_ * step_size_;

    for (int it = 0; it < n_iter; ++it) {
        if (it % 1000 == 0 && debug) {}
        cv::Point2f xrand = (uniform_dist_(rng_) < goal_sample_rate_) ? goal : uniformSample();
        if (!informedFilter(xrand, start, goal, best_cost)) continue;

        int nn_idx = findNearest(xrand, nodes);
        const Node& nearest = nodes[nn_idx];
        cv::Point2f xnew = steer(nearest.pos, xrand);

        int ix = static_cast<int>(xnew.x), iy = static_cast<int>(xnew.y);
        if (ix < 0 || iy < 0 || ix >= w_ || iy >= h_) continue;
        if (occ_.at<uchar>(iy, ix) != 0) continue;
        if (!collisionFree(nearest.pos, xnew)) continue;

        int n = static_cast<int>(nodes.size());
        double radius = rewire_gamma_ * std::sqrt(std::log(n + 1.0) / (n + 1.0));
        radius = std::max(radius, 2.0 * step_size_);
        std::vector<int> near_idxs = findNear(xnew, radius, nodes);

        int best_parent = nn_idx;
        double best_new_cost = nearest.cost + dist(nearest.pos, xnew);
        for (int j : near_idxs) {
            const Node& nj = nodes[j];
            double c = nj.cost + dist(nj.pos, xnew);
            if (c < best_new_cost && collisionFree(nj.pos, xnew)) {
                best_parent = j;
                best_new_cost = c;
            }
        }

        nodes.push_back(Node(xnew, best_parent, best_new_cost));
        int new_idx = static_cast<int>(nodes.size()) - 1;

        for (int j : near_idxs) {
            Node& nj = nodes[j];
            double c_through_new = nodes[new_idx].cost + dist(nodes[new_idx].pos, nj.pos);
            if (c_through_new < nj.cost && collisionFree(nodes[new_idx].pos, nj.pos)) {
                nj.parent = new_idx;
                nj.cost = c_through_new;
            }
        }

        if (sqdist(xnew, goal) <= step2) {
            double total_cost = nodes[new_idx].cost + dist(xnew, goal);
            if (best_cost < 0 || total_cost < best_cost) {
                best_cost = total_cost;
                best_goal_idx = new_idx;
            }
        }
    }

    if (best_goal_idx < 0) return std::vector<cv::Point2i>();

    std::vector<cv::Point2f> path;
    int cur = best_goal_idx;
    while (cur != -1) {
        path.push_back(nodes[cur].pos);
        cur = nodes[cur].parent;
    }
    std::reverse(path.begin(), path.end());
    path.push_back(goal);

    std::vector<cv::Point2f> filtered;
    filtered.push_back(path[0]);
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] != filtered.back()) filtered.push_back(path[i]);
    }

    if (plot) {
        cv::Mat vis = grid_.clone();
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
        for (const Node& n : nodes) {
            if (n.parent != -1)
                cv::line(vis, cv::Point2i(static_cast<int>(n.pos.x), static_cast<int>(n.pos.y)),
                         cv::Point2i(static_cast<int>(nodes[n.parent].pos.x),
                                    static_cast<int>(nodes[n.parent].pos.y)), cv::Scalar(255, 0, 0), 1);
        }
        for (size_t i = 0; i < filtered.size() - 1; ++i)
            cv::line(vis, cv::Point2i(static_cast<int>(filtered[i].x), static_cast<int>(filtered[i].y)),
                     cv::Point2i(static_cast<int>(filtered[i+1].x), static_cast<int>(filtered[i+1].y)),
                     cv::Scalar(0, 0, 255), 2);
        cv::circle(vis, cv::Point2i(static_cast<int>(start.x), static_cast<int>(start.y)), 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(vis, cv::Point2i(static_cast<int>(goal.x), static_cast<int>(goal.y)), 3, cv::Scalar(0, 165, 255), -1);
        cv::imshow("RRT*", vis);
        cv::waitKey(1);
    }

    std::vector<cv::Point2f> resampled_path = resamplePath(filtered, step_size_);
    std::vector<cv::Point2i> result;
    for (size_t i = 1; i + 2 < resampled_path.size(); ++i)
        result.push_back(cv::Point2i(static_cast<int>(resampled_path[i].x),
                                    static_cast<int>(resampled_path[i].y)));
    return result;
}

} // namespace path_planning

#ifndef RRT_PLANNER_HPP
#define RRT_PLANNER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

#include <Eigen/Dense>

namespace path_planning {

struct Node {
    Eigen::Vector2d pos;
    int parent;
    double cost;
    Node() : parent(-1), cost(0.0) {}
    Node(const Eigen::Vector2d& p, int par = -1, double c = 0.0) : pos(p), parent(par), cost(c) {}
};

class RRTPlanner {
public:
    /** Sampling bounds: -1 means use full grid (0..w-1, 0..h-1). */
    RRTPlanner(const cv::Mat& grid_img,
               double step_size = 8.0,
               double goal_sample_rate = 0.05,
               int robot_radius = 5,
               double rewire_gamma = 30.0,
               int sample_col_min = -1,
               int sample_col_max = -1,
               int sample_row_min = -1,
               int sample_row_max = -1);

    /** Plan from start to goal (in grid coords: x=col, y=row). Returns path as grid indices. */
    std::vector<cv::Point2i> plan(const cv::Point2f& start, const cv::Point2f& goal,
                                  int n_iter = 10000, bool debug = false, bool plot = false);

private:
    double sqdist(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    double dist(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    int findNearest(const Eigen::Vector2d& query, const std::vector<Node>& nodes) const;
    std::vector<int> findNear(const Eigen::Vector2d& query, double radius,
                             const std::vector<Node>& nodes) const;
    cv::Mat binaryDilation(const cv::Mat& binary, int kernel_size) const;
    bool collisionFree(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    Eigen::Vector2d uniformSample() const;
    bool informedFilter(const Eigen::Vector2d& x, const Eigen::Vector2d& start,
                       const Eigen::Vector2d& goal, double max_sum_dist) const;
    Eigen::Vector2d steer(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    std::vector<cv::Point2f> resamplePath(const std::vector<cv::Point2f>& path, double ds) const;

    cv::Mat grid_;
    cv::Mat occ_;
    int h_, w_;
    int sample_col_min_, sample_col_max_, sample_row_min_, sample_row_max_;
    double step_size_;
    double goal_sample_rate_;
    double rewire_gamma_;
    mutable std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
};

} // namespace path_planning

#endif

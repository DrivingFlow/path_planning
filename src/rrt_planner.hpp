#ifndef RRT_PLANNER_HPP
#define RRT_PLANNER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

namespace path_planning {

struct Node {
    cv::Point2f pos;
    int parent;
    double cost;
    Node() : parent(-1), cost(0.0) {}
    Node(const cv::Point2f& p, int par = -1, double c = 0.0) : pos(p), parent(par), cost(c) {}
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
    double sqdist(const cv::Point2f& a, const cv::Point2f& b) const;
    double dist(const cv::Point2f& a, const cv::Point2f& b) const;
    int findNearest(const cv::Point2f& query, const std::vector<Node>& nodes) const;
    std::vector<int> findNear(const cv::Point2f& query, double radius,
                             const std::vector<Node>& nodes) const;
    cv::Mat binaryDilation(const cv::Mat& binary, int kernel_size) const;
    bool collisionFree(const cv::Point2f& a, const cv::Point2f& b) const;
    cv::Point2f uniformSample() const;
    bool informedFilter(const cv::Point2f& x, const cv::Point2f& start,
                       const cv::Point2f& goal, double max_sum_dist) const;
    cv::Point2f steer(const cv::Point2f& a, const cv::Point2f& b) const;
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

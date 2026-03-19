#ifndef ASTAR_ENERGY_PLANNER_HPP
#define ASTAR_ENERGY_PLANNER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace path_planning {

/**
 * A* with energy (clearance-based) cost: prefers paths through high-clearance
 * regions. Grid convention: 0 = free, 255 (or non-zero) = obstacle (same as RRT).
 * Plan from start to goal in grid coords (x=col, y=row). Returns path as grid indices.
 */
class AStarEnergyPlanner {
public:
    /** grid_img: 0 = free, 255 = obstacle (same as OccGridBridge / RRT). */
    AStarEnergyPlanner(const cv::Mat& grid_img,
                       int robot_radius_pixels = 5,
                       int corridor_half_width_pixels = 0);

    /**
     * Plan from start to goal (grid coords: x=col, y=row). Returns path as grid indices.
     * beta_valley: valley strength in A* cost (default 0.1).
     * smooth_alpha, smooth_beta, smooth_n_iter: elastic band smoothing (defaults 0.1, 0.2, 50).
     * resample_ds: arc-length step for resampling (default 8.0 pixels).
     */
    std::vector<cv::Point2i> plan(const cv::Point2f& start, const cv::Point2f& goal,
                                  double beta_valley = 0.1,
                                  double smooth_alpha = 0.1,
                                  double smooth_beta = 0.2,
                                  int smooth_n_iter = 50,
                                  double resample_ds = 8.0);

private:
    double heuristic(int ax, int ay, int bx, int by) const;
    bool edgeCorridorFree(int x0, int y0, int x1, int y1) const;
    void buildNeighbors();
    void computeGradient();
    std::vector<cv::Point2f> smoothPath(std::vector<cv::Point2f> path,
                                        double alpha, double beta, int n_iter) const;
    std::vector<cv::Point2f> resamplePath(const std::vector<cv::Point2f>& path, double ds) const;

    cv::Mat grid_;
    cv::Mat dist_;       // distance to nearest obstacle
    cv::Mat clearance_;  // max(0, dist - robot_radius)
    cv::Mat valid_;      // valid for planning (clearance > 0)
    cv::Mat grad_x_;     // gradient of clearance (for smoothing)
    cv::Mat grad_y_;
    int h_, w_;
    int robot_radius_px_;
    int corridor_half_width_px_;
    double required_clearance_px_;
    double max_clearance_;

    struct Neighbor {
        int dx, dy;
        double cost;
    };
    std::vector<Neighbor> neighbors_;
};

} // namespace path_planning

#endif

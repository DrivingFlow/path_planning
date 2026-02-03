#ifndef OCC_GRID_BRIDGE_HPP
#define OCC_GRID_BRIDGE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * Bridge between 3D point clouds (map frame) and 2D occupancy grids.
 * Map and localization are already in a common frame; no leveling/rotation/translation.
 * Provides transform from world coordinates to grid indices and back,
 * so the controller receives real-world (x, y) in map frame, not indices.
 */
class OccGridBridge {
public:
    OccGridBridge() = default;

    /**
     * Load map PCD and compute grid bounds from raw x, y of points (no leveling).
     * Returns true on success.
     */
    bool loadMapPcd(const std::string& pcd_path, double resolution = 0.05);

    /**
     * Load edited static map PNG (white=free, black=obstacle in file).
     * Stored internally as 0=free, 255=obstacle for RRT.
     * Grid dimensions must match those implied by map PCD bounds (w x h).
     */
    bool loadEditedMapPng(const std::string& png_path);

    /**
     * Set bounds explicitly (e.g. from YAML or precomputed). No rotation/translation.
     */
    void setBounds(double x_min, double x_max, double y_min, double y_max, double resolution);

    /** World (map frame) to grid: (x, y) -> (col, row). */
    void worldToGrid(double x, double y, int& col, int& row) const;

    /** Grid (col, row) to world (map frame) (x, y). */
    void gridToWorld(int col, int row, double& x, double& y) const;

    /**
     * Convert a point cloud (Nx3, map frame) to occupancy grid.
     * Points used as-is; z in [z_min, z_max] marks cell occupied.
     * Returns single-channel 0=free, 255=obstacle, same size as static map.
     */
    cv::Mat pointcloudToOccupancyGrid(const std::vector<std::array<float, 3>>& points_xyz,
                                     double z_min = 0.1, double z_max = 2.0) const;

    /**
     * Merge live occupancy grid with static edited map.
     * Both use 0=free, 255=obstacle; result is max (union of obstacles).
     */
    cv::Mat mergeWithStaticMap(const cv::Mat& live_occupancy) const;

    /**
     * Convert path in grid indices (col, row) to world coordinates (x, y) in map frame.
     */
    std::vector<std::array<double, 2>> pathIndicesToWorld(
        const std::vector<cv::Point2i>& path_indices) const;

    /** Get grid dimensions (width, height) from current bounds. */
    int gridWidth() const { return w_; }
    int gridHeight() const { return h_; }

    /** Get bounds in world (map frame). */
    double xMin() const { return x_min_; }
    double xMax() const { return x_max_; }
    double yMin() const { return y_min_; }
    double yMax() const { return y_max_; }
    double resolution() const { return res_; }

    /** Whether the bridge has been initialized with map and bounds. */
    bool isInitialized() const { return initialized_; }

private:
    void computeGridSize();

    double x_min_ = 0, x_max_ = 0, y_min_ = 0, y_max_ = 0;
    double res_ = 0.05;
    int w_ = 0, h_ = 0;

    cv::Mat static_map_;   // 0=free, 255=obstacle, same size as grid
    bool initialized_ = false;
};

#endif

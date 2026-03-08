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
     * Build ego-centered, yaw-normalized 201×201 occupancy grid from points in map frame.
     * Robot at (robot_x, robot_y, robot_yaw); points transformed to ego, kept if d <= 5m and z in [z_min, z_max].
     * Grid origin so that pixel (100, 100) is robot center; 0=free, 255=obstacle.
     * Used for agent-centered model input (training data format).
     */
    static cv::Mat pointcloudToEgoOccupancyGrid201(
        const std::vector<std::array<float, 3>>& points_xyz,
        double robot_x, double robot_y, double robot_yaw,
        double z_min = 0.1, double z_max = 2.0);

    /**
     * Ego grid metadata: size 201×201, resolution and origin for OccupancyGrid message.
     */
    static constexpr int EGO_GRID_SIZE = 201;
    static constexpr double EGO_RADIUS_M = 5.0;
    static double egoGridResolution() { return 2.0 * EGO_RADIUS_M / (EGO_GRID_SIZE - 1); }
    static double egoGridOriginX() { return -EGO_RADIUS_M; }
    static double egoGridOriginY() { return EGO_RADIUS_M; }

    /**
     * Paste a 201×201 ego-frame occupancy grid into the full map grid.
     * Rotates grid by anchor_yaw about its center, then places center at (anchor_x, anchor_y) in map.
     * map_out must be pre-allocated (e.g. zeros or copy of static); non-free ego cells are written (max with existing).
     */
    void pasteEgoGridIntoMap(const cv::Mat& ego_grid,
                             double anchor_x, double anchor_y, double anchor_yaw,
                             cv::Mat& map_out) const;

    /**
     * Zero out the map region covered by the ego footprint (circle of radius EGO_RADIUS_M
     * centred on anchor_x, anchor_y).  Rotation-independent: iterates over map pixels in
     * the bounding box and zeroes every cell whose world position is within the radius,
     * guaranteeing no map cell inside the circle is missed due to sub-pixel rotation gaps.
     */
    void zeroEgoFootprintInMap(double anchor_x, double anchor_y, double anchor_yaw,
                               cv::Mat& map_out) const;

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

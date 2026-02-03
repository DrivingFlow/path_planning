#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// -----------------------------
// Node container
// -----------------------------
struct Node {
    cv::Point2f pos;
    int parent;
    double cost;
    
    Node() : parent(-1), cost(0.0) {}
    Node(const cv::Point2f& p, int par = -1, double c = 0.0) 
        : pos(p), parent(par), cost(c) {}
};

// -----------------------------
// RRT* Planner
// -----------------------------
class RRTPlanner {
private:
    cv::Mat grid;
    cv::Mat occ;  // Inflated obstacles
    int h, w;
    double step_size;
    double goal_sample_rate;
    double rewire_gamma;
    
    mutable std::mt19937 rng;  // mutable allows modification in const methods
    std::uniform_real_distribution<double> uniform_dist;
    
    // Utility functions
    double sqdist(const cv::Point2f& a, const cv::Point2f& b) const {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return dx * dx + dy * dy;
    }
    
    double dist(const cv::Point2f& a, const cv::Point2f& b) const {
        return std::sqrt(sqdist(a, b));
    }
    
    // Simple nearest neighbor search (linear search for now)
    int findNearest(const cv::Point2f& query, const std::vector<Node>& nodes) const {
        int best_idx = 0;
        double best_dist = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            double d = sqdist(query, nodes[i].pos);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        return best_idx;
    }
    
    // Find all nodes within radius
    std::vector<int> findNear(const cv::Point2f& query, double radius, 
                              const std::vector<Node>& nodes) const {
        std::vector<int> near_idxs;
        double radius_sq = radius * radius;
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (sqdist(query, nodes[i].pos) <= radius_sq) {
                near_idxs.push_back(i);
            }
        }
        return near_idxs;
    }
    
    // Binary dilation using OpenCV
    cv::Mat binaryDilation(const cv::Mat& binary, int kernel_size) const {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                    cv::Size(kernel_size, kernel_size));
        cv::Mat dilated;
        cv::dilate(binary, dilated, kernel);
        return dilated;
    }

public:
    RRTPlanner(const cv::Mat& grid_img, 
               double step_size = 8.0,
               double goal_sample_rate = 0.05,
               int robot_radius = 5,
               double rewire_gamma = 30.0)
        : grid(grid_img.clone()),
          step_size(step_size),
          goal_sample_rate(goal_sample_rate),
          rewire_gamma(rewire_gamma),
          rng(std::random_device{}()),
          uniform_dist(0.0, 1.0) {
        
        h = grid.rows;
        w = grid.cols;
        
        // Inflate obstacles ONCE (huge speed win)
        cv::Mat binary = grid >= 50;
        int kernel_size = 2 * robot_radius + 1;
        occ = binaryDilation(binary, kernel_size);
    }
    
    // -------------------------
    // Collision checking
    // -------------------------
    bool collisionFree(const cv::Point2f& a, const cv::Point2f& b) const {
        double length = dist(a, b);
        int n = static_cast<int>(length / 3.0) + 1;
        
        for (int i = 0; i <= n; ++i) {
            double t = static_cast<double>(i) / n;
            int x = static_cast<int>(a.x + t * (b.x - a.x));
            int y = static_cast<int>(a.y + t * (b.y - a.y));
            
            if (x < 0 || y < 0 || x >= w || y >= h) {
                return false;
            }
            if (occ.at<uchar>(y, x) != 0) {
                return false;
            }
        }
        return true;
    }
    
    // -------------------------
    // Sampling
    // -------------------------
    cv::Point2f uniformSample() const {
        std::uniform_real_distribution<double> x_dist(0.0, w);
        std::uniform_real_distribution<double> y_dist(0.0, h);
        return cv::Point2f(x_dist(rng), y_dist(rng));
    }
    
    bool informedFilter(const cv::Point2f& x, const cv::Point2f& start, 
                       const cv::Point2f& goal, double max_sum_dist) const {
        if (max_sum_dist < 0) {  // None/null check
            return true;
        }
        return dist(x, start) + dist(x, goal) <= max_sum_dist;
    }
    
    // -------------------------
    // Steering
    // -------------------------
    cv::Point2f steer(const cv::Point2f& a, const cv::Point2f& b) const {
        cv::Point2f v = b - a;
        double d = cv::norm(v);
        if (d <= step_size) {
            return b;
        }
        return a + (v / d) * step_size;
    }
    
    std::vector<cv::Point2f> resamplePath(const std::vector<cv::Point2f>& path, double ds) const {
        /**
         * Resample a 2D polyline path with uniform spacing ds.
         */
        if (path.size() < 2) {
            return path;
        }
        
        // Compute cumulative arc length
        std::vector<double> seg_lengths;
        std::vector<double> s;
        s.push_back(0.0);
        
        for (size_t i = 0; i < path.size() - 1; ++i) {
            double seg_len = dist(path[i], path[i + 1]);
            seg_lengths.push_back(seg_len);
            s.push_back(s.back() + seg_len);
        }
        
        double total_length = s.back();
        
        if (total_length < ds) {
            return {path[0], path.back()};
        }
        
        // New sample positions
        std::vector<double> s_new;
        for (double si = 0.0; si < total_length; si += ds) {
            s_new.push_back(si);
        }
        s_new.push_back(total_length);
        
        // Interpolate
        std::vector<cv::Point2f> new_pts;
        size_t seg_idx = 0;
        
        for (double si : s_new) {
            while (seg_idx < seg_lengths.size() - 1 && s[seg_idx + 1] < si) {
                seg_idx++;
            }
            
            double seg_len = std::max(seg_lengths[seg_idx], 1e-9);
            double t = (si - s[seg_idx]) / seg_len;
            cv::Point2f p = path[seg_idx] + t * (path[seg_idx + 1] - path[seg_idx]);
            new_pts.push_back(p);
        }
        
        return new_pts;
    }
    
    // -------------------------
    // Plan
    // -------------------------
    std::vector<cv::Point2i> plan(const cv::Point2f& start, const cv::Point2f& goal, 
                                  int n_iter = 10000, bool debug = false, bool plot = false) {
        std::vector<Node> nodes;
        nodes.push_back(Node(start, -1, 0.0));
        
        int best_goal_idx = -1;
        double best_cost = -1.0;
        double step2 = step_size * step_size;
        
        for (int it = 0; it < n_iter; ++it) {
            if (it % 1000 == 0 && debug) {
                std::cout << "Iteration " << it << std::endl;
            }
            
            // ---------------------
            // Sample
            // ---------------------
            cv::Point2f xrand;
            if (uniform_dist(rng) < goal_sample_rate) {
                xrand = goal;
            } else {
                xrand = uniformSample();
            }
            
            // ellipse filter ONLY for new samples
            if (!informedFilter(xrand, start, goal, best_cost)) {
                continue;
            }
            
            // ---------------------
            // Nearest
            // ---------------------
            int nn_idx = findNearest(xrand, nodes);
            const Node& nearest = nodes[nn_idx];
            
            cv::Point2f xnew = steer(nearest.pos, xrand);
            
            int ix = static_cast<int>(xnew.x);
            int iy = static_cast<int>(xnew.y);
            if (ix < 0 || iy < 0 || ix >= w || iy >= h) {
                continue;
            }
            if (occ.at<uchar>(iy, ix) != 0) {
                continue;
            }
            if (!collisionFree(nearest.pos, xnew)) {
                continue;
            }
            
            // ---------------------
            // Near (rewiring)
            // ---------------------
            int n = nodes.size();
            double radius = rewire_gamma * std::sqrt(std::log(n + 1.0) / (n + 1.0));
            radius = std::max(radius, 2.0 * step_size);
            
            std::vector<int> near_idxs = findNear(xnew, radius, nodes);
            
            // choose best parent
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
            
            Node new_node(xnew, best_parent, best_new_cost);
            nodes.push_back(new_node);
            
            int new_idx = nodes.size() - 1;
            
            // rewire neighbors
            for (int j : near_idxs) {
                Node& nj = nodes[j];
                double c_through_new = new_node.cost + dist(new_node.pos, nj.pos);
                if (c_through_new < nj.cost && collisionFree(new_node.pos, nj.pos)) {
                    nj.parent = new_idx;
                    nj.cost = c_through_new;
                }
            }
            
            // ---------------------
            // Goal check
            // ---------------------
            if (sqdist(xnew, goal) <= step2) {
                double total_cost = new_node.cost + dist(xnew, goal);
                if (best_cost < 0 || total_cost < best_cost) {
                    best_cost = total_cost;
                    best_goal_idx = new_idx;
                }
            }
        }
        
        // -------------------------
        // Reconstruct path
        // -------------------------
        if (best_goal_idx < 0) {
            return std::vector<cv::Point2i>();
        }
        
        std::vector<cv::Point2f> path;
        int cur = best_goal_idx;
        while (cur != -1) {
            path.push_back(nodes[cur].pos);
            cur = nodes[cur].parent;
        }
        std::reverse(path.begin(), path.end());
        path.push_back(goal);
        
        // remove consecutive duplicates
        std::vector<cv::Point2f> filtered;
        filtered.push_back(path[0]);
        for (size_t i = 1; i < path.size(); ++i) {
            if (path[i] != filtered.back()) {
                filtered.push_back(path[i]);
            }
        }
        
        // -------------------------
        // Optional plot
        // -------------------------
        if (plot) {
            cv::Mat vis = grid.clone();
            cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
            
            // Draw tree
            for (const Node& n : nodes) {
                if (n.parent != -1) {
                    cv::line(vis, 
                            cv::Point2i(static_cast<int>(n.pos.x), static_cast<int>(n.pos.y)),
                            cv::Point2i(static_cast<int>(nodes[n.parent].pos.x), 
                                       static_cast<int>(nodes[n.parent].pos.y)),
                            cv::Scalar(255, 0, 0), 1);
                }
            }
            
            // Draw path
            for (size_t i = 0; i < filtered.size() - 1; ++i) {
                cv::line(vis,
                        cv::Point2i(static_cast<int>(filtered[i].x), static_cast<int>(filtered[i].y)),
                        cv::Point2i(static_cast<int>(filtered[i+1].x), static_cast<int>(filtered[i+1].y)),
                        cv::Scalar(0, 0, 255), 2);
            }
            
            // Draw start and goal
            cv::circle(vis, cv::Point2i(static_cast<int>(start.x), static_cast<int>(start.y)), 
                      3, cv::Scalar(0, 255, 0), -1);
            cv::circle(vis, cv::Point2i(static_cast<int>(goal.x), static_cast<int>(goal.y)), 
                      3, cv::Scalar(0, 165, 255), -1);
            
            cv::imshow("Optimized RRT*", vis);
            cv::waitKey(0);
        }
        
        // Uniformly resample path
        double spacing = step_size;  // or smaller if you want smoother tracking
        std::vector<cv::Point2f> resampled_path = resamplePath(filtered, spacing);
        
        // Convert to integer points, skip first and last two
        std::vector<cv::Point2i> result;
        for (size_t i = 1; i < resampled_path.size() - 2; ++i) {
            result.push_back(cv::Point2i(static_cast<int>(resampled_path[i].x),
                                        static_cast<int>(resampled_path[i].y)));
        }
        
        return result;
    }
};

// ============================================================
//  INTERACTIVE POINT SELECTION
// ============================================================

struct PointSelector {
    std::vector<cv::Point2f> points;
    cv::Mat display_img;
    bool done;
    
    PointSelector() : done(false) {}
    
    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        PointSelector* selector = static_cast<PointSelector*>(userdata);
        
        if (event == cv::EVENT_LBUTTONDOWN && selector->points.size() < 2) {
            selector->points.push_back(cv::Point2f(x, y));
            
            // Draw the point on the image
            cv::Scalar color = (selector->points.size() == 1) ? 
                cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);  // Green for start, Orange for goal
            cv::circle(selector->display_img, cv::Point(x, y), 5, color, -1);
            
            std::string label = (selector->points.size() == 1) ? "Start" : "Goal";
            cv::putText(selector->display_img, label, 
                       cv::Point(x + 10, y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            
            cv::imshow("Click start then goal", selector->display_img);
            
            if (selector->points.size() == 2) {
                selector->done = true;
                std::cout << "Start: (" << selector->points[0].x << ", " << selector->points[0].y << ")" << std::endl;
                std::cout << "Goal: (" << selector->points[1].x << ", " << selector->points[1].y << ")" << std::endl;
            }
        }
    }
};

// ============================================================
//  MAIN FUNCTION
// ============================================================

int main(int argc, char** argv) {
    try {
        std::cout << "Program started!" << std::endl;
        std::cerr << "Program started (stderr)!" << std::endl;
        std::cout.flush();
        std::cerr.flush();
        
        // ============================================================
        //  LOAD PNG MAP
        // ============================================================
        
        std::cout << "Attempting to load plab_raw2.png..." << std::endl;
        std::cout.flush();
        
        cv::Mat grid = cv::imread("plab_raw2.png", cv::IMREAD_GRAYSCALE);
        if (grid.empty()) {
        std::cerr << "Error: Could not load image plab_raw2.png" << std::endl;
        std::cerr << "Current working directory: ";
        #ifdef _WIN32
        char cwd[1024];
        if (GetCurrentDirectoryA(1024, cwd)) {
            std::cerr << cwd << std::endl;
        }
        #else
        char* cwd = getcwd(NULL, 0);
        if (cwd) {
            std::cerr << cwd << std::endl;
            free(cwd);
        }
        #endif
        std::cerr << "Make sure plab_raw2.png is in the same directory as the executable." << std::endl;
        return -1;
    }
    
    grid = 255 - grid;  // invert (white free, black obstacles)
    std::cout << "Grid shape (H,W) = (" << grid.rows << ", " << grid.cols << ")" << std::endl;
    std::cout.flush();
    
    // ============================================================
    //  CHOOSE START AND GOAL
    // ============================================================
    
    cv::Point2f start, goal;
    
    if (argc >= 5) {
        // Use command line arguments if provided
        start = cv::Point2f(std::stof(argv[1]), std::stof(argv[2]));
        goal = cv::Point2f(std::stof(argv[3]), std::stof(argv[4]));
        std::cout << "Start: (" << start.x << ", " << start.y << ")" << std::endl;
        std::cout << "Goal: (" << goal.x << ", " << goal.y << ")" << std::endl;
    } else {
        // Interactive point selection (like Python version)
        std::cout << "Click on the image to select start and goal points" << std::endl;
        std::cout << "1. Click once for START point (green)" << std::endl;
        std::cout << "2. Click again for GOAL point (orange)" << std::endl;
        std::cout << "Waiting for window to open..." << std::endl;
        std::cout.flush();
        
        cv::Mat display_img;
        cv::cvtColor(255 - grid, display_img, cv::COLOR_GRAY2BGR);
        
        PointSelector selector;
        selector.display_img = display_img.clone();
        
        // Check if OpenCV GUI is available
        std::cout << "Attempting to create OpenCV window..." << std::endl;
        std::cout.flush();
        
        cv::namedWindow("Click start then goal", cv::WINDOW_NORMAL);
        cv::setMouseCallback("Click start then goal", PointSelector::onMouse, &selector);
        cv::imshow("Click start then goal", selector.display_img);
        
        std::cout << "Window created. If you don't see a window, OpenCV GUI may not work in this environment." << std::endl;
        std::cout << "Waiting 5 seconds to see if window appears..." << std::endl;
        std::cout.flush();
        
        // Give user time to see if window appears
        for (int i = 5; i > 0 && !selector.done; --i) {
            cv::waitKey(1000);
            std::cout << "  " << i << "..." << std::endl;
            std::cout.flush();
        }
        
        if (!selector.done) {
            std::cout << "No window detected. Using default start/goal positions." << std::endl;
            std::cout << "To specify points, use: ./occupancy_grid_planning_opt.exe start_x start_y goal_x goal_y" << std::endl;
            std::cout.flush();
            cv::destroyAllWindows();
            // Use default positions
            start = cv::Point2f(100, 100);
            goal = cv::Point2f(500, 500);
        } else {
            std::cout << "Window should be visible now. Click on the image..." << std::endl;
            std::cout.flush();
            
            // Wait for two clicks
            int wait_count = 0;
            while (!selector.done) {
                int key = cv::waitKey(30) & 0xFF;
                if (key == 27) {  // ESC to cancel
                    std::cout << "Cancelled by user" << std::endl;
                    cv::destroyAllWindows();
                    return 0;
                }
                wait_count++;
                // Print progress every 100 iterations (3 seconds) to show it's waiting
                if (wait_count % 100 == 0) {
                    std::cout << "Waiting for clicks... (" << selector.points.size() << "/2 points selected)" << std::endl;
                    std::cout.flush();
                }
            }
            
            cv::destroyWindow("Click start then goal");
            
            start = selector.points[0];
            goal = selector.points[1];
        }
    }
    
    // ============================================================
    //  RUN RRT PLANNER
    // ============================================================
    
    std::cout << "Starting path planning..." << std::endl;
    std::cout.flush();
    
    RRTPlanner planner(grid, 8.0, 0.05, 5, 30.0);
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point2i> path = planner.plan(start, goal, 10000, true);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Planned path in " << duration.count() / 1000.0 << "s" << std::endl;
    std::cout << "Path points: " << path.size() << std::endl;
    std::cout.flush();
    
    if (path.empty()) {
        std::cerr << "Error: No path found!" << std::endl;
        std::cerr << "Try different start/goal positions or increase iterations." << std::endl;
        return -1;
    }
    
    // ============================================================
    //  DRAW RESULT ON IMAGE
    // ============================================================
    
    cv::Mat img;
    cv::cvtColor(255 - grid, img, cv::COLOR_GRAY2BGR);
    
    for (const cv::Point2i& pt : path) {
        cv::rectangle(img, 
                     cv::Rect(pt.x - 1, pt.y - 1, 3, 3),
                     cv::Scalar(0, 0, 255), -1);  // red
    }
    
    cv::Point2i start_int(static_cast<int>(start.x), static_cast<int>(start.y));
    cv::Point2i goal_int(static_cast<int>(goal.x), static_cast<int>(goal.y));
    cv::rectangle(img, 
                 cv::Rect(start_int.x - 1, start_int.y - 1, 3, 3),
                 cv::Scalar(0, 255, 0), -1);  // green start
    cv::rectangle(img,
                 cv::Rect(goal_int.x - 1, goal_int.y - 1, 3, 3),
                 cv::Scalar(255, 0, 0), -1);  // blue goal
    
    std::cout << "Displaying result..." << std::endl;
    std::cout.flush();
    
    // Save image as fallback
    cv::imwrite("planned_path_result.png", img);
    std::cout << "Result saved to: planned_path_result.png" << std::endl;
    
    cv::namedWindow("Planned Path", cv::WINDOW_NORMAL);
    cv::imshow("Planned Path", img);
    std::cout << "Window opened. Press any key in the window to close, or close the window." << std::endl;
    std::cout.flush();
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    std::cout << "Done!" << std::endl;
    return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
        return -1;
    }
}

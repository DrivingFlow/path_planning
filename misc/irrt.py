import numpy as np
import math
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import cv2
from time import time

# -----------------------------
# Utility
# -----------------------------
def sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def dist(a, b):
    return math.sqrt(sqdist(a, b))

# -----------------------------
# Node container (C++ style)
# -----------------------------
class Node:
    __slots__ = ("pos", "parent", "cost")
    def __init__(self, pos, parent=-1, cost=0.0):
        self.pos = pos
        self.parent = parent
        self.cost = cost

# -----------------------------
# RRT* Planner
# -----------------------------
class RRTPlanner:
    def __init__(
        self,
        grid,
        step_size=8.0,
        goal_sample_rate=0.05,
        robot_radius=5,
        rewire_gamma=30.0
    ):
        self.grid = grid
        self.h, self.w = grid.shape
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.rewire_gamma = rewire_gamma

        # Inflate obstacles ONCE (huge speed win)
        struct = np.ones((2*robot_radius+1, 2*robot_radius+1), dtype=bool)
        self.occ = binary_dilation(grid >= 50, structure=struct)

    # -------------------------
    # Collision checking
    # -------------------------
    def collision_free(self, a, b):
        length = dist(a, b)
        n = int(length / 3) + 1
        for i in range(n + 1):
            t = i / n
            x = int(a[0] + t * (b[0] - a[0]))
            y = int(a[1] + t * (b[1] - a[1]))
            if x < 0 or y < 0 or x >= self.w or y >= self.h:
                return False
            if self.occ[y, x]:
                return False
        return True

    # -------------------------
    # Sampling
    # -------------------------
    def uniform_sample(self):
        return np.array([
            np.random.uniform(0, self.w),
            np.random.uniform(0, self.h)
        ])

    def informed_filter(self, x, start, goal, max_sum_dist):
        if max_sum_dist is None:
            return True
        return dist(x, start) + dist(x, goal) <= max_sum_dist

    # -------------------------
    # Steering
    # -------------------------
    def steer(self, a, b):
        v = b - a
        d = np.linalg.norm(v)
        if d <= self.step_size:
            return b
        return a + (v / d) * self.step_size

    def resample_path(self, path, ds):
        """
        Resample a 2D polyline path with uniform spacing ds.
        """
        if len(path) < 2:
            return path

        # Convert to float arrays
        pts = np.array(path, dtype=float)

        # Compute cumulative arc length
        deltas = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = s[-1]

        if total_length < ds:
            return [tuple(pts[0]), tuple(pts[-1])]

        # New sample positions
        s_new = np.arange(0, total_length, ds)
        s_new = np.append(s_new, total_length)

        # Interpolate
        new_pts = []
        seg_idx = 0

        for si in s_new:
            while seg_idx < len(seg_lengths) - 1 and s[seg_idx + 1] < si:
                seg_idx += 1

            t = (si - s[seg_idx]) / max(seg_lengths[seg_idx], 1e-9)
            p = pts[seg_idx] + t * deltas[seg_idx]
            new_pts.append(tuple(p))

        return new_pts


    # -------------------------
    # Plan
    # -------------------------
    def plan(self, start, goal, n_iter=10000, debug=False, plot=False):
        start = np.array(start, dtype=float)
        goal  = np.array(goal, dtype=float)

        nodes = [Node(start, parent=-1, cost=0.0)]
        positions = [start]
        kd_tree = KDTree(positions)

        best_goal_idx = None
        best_cost = None
        step2 = self.step_size ** 2

        for it in range(n_iter):
            if it % 1000 == 0:
                print(f"Iteration {it}")
            # ---------------------
            # Sample
            # ---------------------
            if np.random.rand() < self.goal_sample_rate:
                xrand = goal
            else:
                xrand = self.uniform_sample()

            # ellipse filter ONLY for new samples
            if not self.informed_filter(xrand, start, goal, best_cost):
                continue

            # ---------------------
            # Nearest
            # ---------------------
            _, nn_idx = kd_tree.query(xrand)
            nearest = nodes[nn_idx]

            xnew = self.steer(nearest.pos, xrand)

            ix, iy = int(xnew[0]), int(xnew[1])
            if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
                continue
            if self.occ[iy, ix]:
                continue
            if not self.collision_free(nearest.pos, xnew):
                continue

            # ---------------------
            # Near (rewiring)
            # ---------------------
            n = len(nodes)
            radius = self.rewire_gamma * math.sqrt(math.log(n+1)/(n+1))
            radius = max(radius, 2 * self.step_size)

            near_idxs = kd_tree.query_ball_point(xnew, radius)

            # choose best parent
            best_parent = nn_idx
            best_new_cost = nearest.cost + dist(nearest.pos, xnew)

            for j in near_idxs:
                nj = nodes[j]
                c = nj.cost + dist(nj.pos, xnew)
                if c < best_new_cost and self.collision_free(nj.pos, xnew):
                    best_parent = j
                    best_new_cost = c

            new_node = Node(xnew, parent=best_parent, cost=best_new_cost)
            nodes.append(new_node)
            positions.append(xnew)

            # rebuild KD-tree lazily
            if len(nodes) % 25 == 0:
                kd_tree = KDTree(positions)

            new_idx = len(nodes) - 1

            # rewire neighbors
            for j in near_idxs:
                nj = nodes[j]
                c_through_new = new_node.cost + dist(new_node.pos, nj.pos)
                if c_through_new < nj.cost and self.collision_free(new_node.pos, nj.pos):
                    nj.parent = new_idx
                    nj.cost = c_through_new

            # ---------------------
            # Goal check
            # ---------------------
            if sqdist(xnew, goal) <= step2:
                total_cost = new_node.cost + dist(xnew, goal)
                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_goal_idx = new_idx

        # -------------------------
        # Reconstruct path
        # -------------------------
        if best_goal_idx is None:
            return None

        path = []
        cur = best_goal_idx
        while cur != -1:
            path.append(tuple(nodes[cur].pos))
            cur = nodes[cur].parent
        path.reverse()
        path.append(tuple(goal))

        # remove consecutive duplicates
        filtered = [path[0]]
        for p in path[1:]:
            if p != filtered[-1]:
                filtered.append(p)

        # -------------------------
        # Optional plot
        # -------------------------
        if plot:
            plt.imshow(self.grid, cmap="gray_r")
            for i, n in enumerate(nodes):
                if n.parent != -1:
                    p = nodes[n.parent]
                    plt.plot(
                        [n.pos[0], p.pos[0]],
                        [n.pos[1], p.pos[1]],
                        "b-", linewidth=0.3
                    )
            px = [p[0] for p in filtered]
            py = [p[1] for p in filtered]
            plt.plot(px, py, "r-", linewidth=2)
            plt.scatter(start[0], start[1], c="green", s=40)
            plt.scatter(goal[0], goal[1], c="orange", s=40)
            plt.title("Optimized RRT*")
            plt.axis("equal")
            plt.show()

        # Uniformly resample path
        spacing = self.step_size  # or smaller if you want smoother tracking
        resampled_path = self.resample_path(filtered, spacing)

        return [(int(x), int(y)) for x, y in resampled_path[1:-2]]




'''
    Pseudo code for how things should run on the dog

    1. Initialize: read plab pcd, get bounds and read editted plab OC
    2. either prompt user to select endpoint OR wait for requested end point
    3. Iteravely, plan path, convert to map frame, send way points.
        a. Everytime new livescan is given, get new OC and check if current path intersects anything occupied (make sure radius taken into account)
            i. If so, back to step 3
            ii. if not, back to 3a
        b. Wait for indicator from controller that destination is reached and stop planning.
'''



# ============================================================
#  LOAD PNG MAP
# ============================================================

grid = cv2.imread("plab_raw2.png", cv2.IMREAD_GRAYSCALE)
grid = 255 - grid   # invert (white free, black obstacles)
grid_np = np.array(grid)
print("Grid shape (H,W) =", grid_np.shape)

# ============================================================
#  CHOOSE START AND GOAL
# ============================================================

plt.imshow(255-grid, cmap="gray")
plt.title("Click start then goal")
pts = plt.ginput(2)
plt.close()

start = tuple(map(int, pts[0]))  # x,y
goal  = tuple(map(int, pts[1]))  # x,y
print("Start:", start)
print("Goal:", goal)

# ============================================================
#  RUN RRT PLANNER
# ============================================================

planner = RRTPlanner(grid, robot_radius=5)
start_time = time()
path = planner.plan(start, goal)
print(f'Planned path in {time()-start_time}s')
print("Path points:", len(path))

# ============================================================
#  DRAW RESULT ON IMAGE
# ============================================================

img = np.stack([255-grid, 255-grid, 255-grid], axis=-1).copy()

for (x, y) in path:
    img[y-1:y+1, x-1:x+1] = (0, 0, 255)  # red

sx, sy = start
gx, gy = goal
img[sy-1:sy+1, sx-1:sx+1] = (0, 255, 0)  # green start
img[gy-1:gy+1, gx-1:gx+1] = (255, 0, 0)  # blue goal

plt.figure(figsize=(10, 12))
plt.imshow(img)
plt.title("Planned Path")
plt.show()

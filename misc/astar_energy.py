import numpy as np
import cv2
import heapq
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from time import time


class EnergyAStarPlanner:
    def __init__(self, binary_grid, robot_radius_pixels=5):

        self.binary = binary_grid
        self.h, self.w = binary_grid.shape

        self.obstacles = binary_grid > 0

        # Distance transform
        self.dist = distance_transform_edt(~self.obstacles)

        self.robot_radius_pixels = robot_radius_pixels
        self.valid = self.dist > robot_radius_pixels

        # Normalized clearance
        eps = 1e-6
        clearance = np.maximum(self.dist - robot_radius_pixels, eps)
        self.clearance = clearance
        self.max_clearance = np.max(clearance)

        # 16-connected neighbors (reduces grid bias)
        self.neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                self.neighbors.append((dx, dy, np.hypot(dx, dy)))

        # add half diagonals
        for dx, dy in [(2,1),(1,2),(-2,1),(-1,2),(2,-1),(1,-2),(-2,-1),(-1,-2)]:
            self.neighbors.append((dx, dy, np.hypot(dx, dy)))

        # Precompute gradient of clearance for smoothing
        self.grad_y, self.grad_x = np.gradient(self.clearance)

    # --------------------------------------------------------
    # Heuristic
    # --------------------------------------------------------

    def heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # --------------------------------------------------------
    # A* Planning
    # --------------------------------------------------------

    def plan(self, start, goal):

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        if not self.valid[start[1], start[0]]:
            print("Start in collision.")
            return None
        if not self.valid[goal[1], goal[0]]:
            print("Goal in collision.")
            return None

        g_score = np.full((self.h, self.w), np.inf)
        parent = np.full((self.h, self.w, 2), -1, dtype=int)
        visited = np.zeros((self.h, self.w), dtype=bool)

        pq = []
        g_score[start[1], start[0]] = 0
        heapq.heappush(pq, (0, start))

        beta = 0.1  # valley strength

        while pq:
            _, current = heapq.heappop(pq)
            cx, cy = current

            if visited[cy, cx]:
                continue
            visited[cy, cx] = True

            if current == goal:
                break

            for dx, dy, move_cost in self.neighbors:

                nx = cx + dx
                ny = cy + dy

                if nx < 0 or ny < 0 or nx >= self.w or ny >= self.h:
                    continue
                if not self.valid[ny, nx]:
                    continue

                # Exponential valley preference
                clearance = self.clearance[ny, nx]
                norm_clear = clearance / self.max_clearance

                energy_factor = np.exp(-beta * norm_clear)

                cost = move_cost * energy_factor

                tentative_g = g_score[cy, cx] + cost

                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    f = tentative_g + self.heuristic((nx, ny), goal)
                    heapq.heappush(pq, (f, (nx, ny)))
                    parent[ny, nx] = [cx, cy]

        if g_score[goal[1], goal[0]] == np.inf:
            print("No path found.")
            return None

        # Reconstruct path
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            px, py = parent[cur[1], cur[0]]
            cur = (px, py)
        path.append(start)
        path.reverse()

        return path

    # --------------------------------------------------------
    # High-Resolution Line Collision
    # --------------------------------------------------------

    def line_collision_free(self, a, b):

        x0, y0 = a
        x1, y1 = b

        length = int(np.hypot(x1-x0, y1-y0))
        samples = max(length * 3, 10)

        for t in np.linspace(0, 1, samples):
            x = int(x0 + t*(x1-x0))
            y = int(y0 + t*(y1-y0))
            if not self.valid[y, x]:
                return False
        return True

    # --------------------------------------------------------
    # Elastic Band Smoothing
    # --------------------------------------------------------

    def smooth_path(self, path, alpha=0.1, beta=0.2, n_iter=0):

        if path is None or len(path) < 3:
            return path

        path = np.array(path, dtype=float)

        for _ in range(n_iter):
            new_path = path.copy()

            for i in range(1, len(path)-1):

                prev_pt = path[i-1]
                next_pt = path[i+1]
                curr = path[i]

                # Smoothness force
                smooth_force = alpha * (prev_pt + next_pt - 2*curr)

                # Obstacle repulsion (push toward high clearance)
                x, y = int(curr[0]), int(curr[1])
                grad = np.array([self.grad_x[y, x], self.grad_y[y, x]])

                repel_force = beta * grad

                new_pt = curr + smooth_force + repel_force

                xi, yi = int(new_pt[0]), int(new_pt[1])
                if 0 <= xi < self.w and 0 <= yi < self.h:
                    if self.valid[yi, xi]:
                        new_path[i] = new_pt

            path = new_path

        return [tuple(p) for p in path]

    # --------------------------------------------------------
    # Uniform Arc-Length Resampling
    # --------------------------------------------------------

    def resample_path(self, path, ds):

        if path is None or len(path) < 2:
            return path

        pts = np.array(path, dtype=float)
        deltas = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)

        s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = s[-1]

        if total_length < ds:
            return [tuple(pts[0]), tuple(pts[-1])]

        s_new = np.arange(0, total_length, ds)
        s_new = np.append(s_new, total_length)

        new_pts = []
        seg_idx = 0

        for si in s_new:
            while seg_idx < len(seg_lengths)-1 and s[seg_idx+1] < si:
                seg_idx += 1

            t = (si - s[seg_idx]) / max(seg_lengths[seg_idx], 1e-9)
            p = pts[seg_idx] + t * deltas[seg_idx]
            new_pts.append(tuple(p))

        return new_pts



# ============================================================
# LOAD MAP
# ============================================================

grid = cv2.imread("plab_4-1_rotated_crop.png", cv2.IMREAD_GRAYSCALE)
grid = 255 - grid
grid = np.array(grid)

print("Grid shape:", grid.shape)

# ============================================================
# CLICK START + GOAL
# ============================================================

# plt.imshow(255-grid, cmap="gray")
# plt.title("Click START then GOAL")
# pts = plt.ginput(2)
# plt.close()

# start = tuple(map(int, pts[0]))
# goal = tuple(map(int, pts[1]))


start = (69, 473)
goal = (277, 68)
print("Start:", start)
print("Goal:", goal)

# ============================================================
# PLAN
# ============================================================

planner = EnergyAStarPlanner(grid, robot_radius_pixels=5)

t0 = time()
path_unsmoothed = planner.plan(start, goal)
print("Planning time:", time() - t0)

# ============================================================
# LOOP OVER N_ITER VALUES FOR SMOOTHING COMPARISON
# ============================================================

n_iter_values = [50]
smoothed_paths = {}

for n_iter in n_iter_values:
    path = planner.smooth_path(path_unsmoothed.copy() if path_unsmoothed else None, n_iter=n_iter)
    step_size = 8.0
    path = planner.resample_path(path, step_size)
    smoothed_paths[n_iter] = path

# ============================================================
# VISUALIZE
# ============================================================

# Colors for different n_iter values
colors = plt.cm.tab20(np.linspace(0, 1, len(n_iter_values)))

plt.figure(figsize=(14, 10))

plt.subplot(1, 2, 1)
plt.imshow(255 - grid, cmap="gray")
plt.title("Binary Grid + Smoothed Paths (different n_iter)")

# Plot all paths
for idx, n_iter in enumerate(n_iter_values):
    path = smoothed_paths[n_iter]
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    plt.plot(px, py, color=colors[idx], label=f"n_iter={n_iter}", marker='o', markersize=2, linewidth=1.5)

sx, sy = start
gx, gy = goal
plt.scatter(start[0], start[1], c='lime', s=100, marker='s', edgecolors='black', linewidth=2, label='Start', zorder=5)
plt.scatter(goal[0], goal[1], c='red', s=100, marker='^', edgecolors='black', linewidth=2, label='Goal', zorder=5)
plt.legend(loc='upper left', fontsize=8, ncol=2)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.imshow(planner.dist, cmap="viridis")
plt.title("Distance Transform + Smoothed Paths")

# Plot all paths on distance transform
for idx, n_iter in enumerate(n_iter_values):
    path = smoothed_paths[n_iter]
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    plt.plot(px, py, color=colors[idx], label=f"n_iter={n_iter}", marker='o', markersize=2, linewidth=1.5)

plt.scatter(start[0], start[1], c='lime', s=100, marker='s', edgecolors='black', linewidth=2, label='Start', zorder=5)
plt.scatter(goal[0], goal[1], c='red', s=100, marker='^', edgecolors='black', linewidth=2, label='Goal', zorder=5)
plt.legend(loc='upper left', fontsize=8, ncol=2)
plt.axis('equal')

plt.tight_layout()
plt.show()

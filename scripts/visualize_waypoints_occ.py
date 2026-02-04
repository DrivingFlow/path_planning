#!/usr/bin/env python3
"""
Real-time visualization of occupancy grid and waypoints for the path planner.
Subscribes to placeholder topics; adjust topic names to match your ROS2 setup.
"""

import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped


class WaypointsOccVisualizer(Node):
    def __init__(self, occ_topic, path_topic, waypoints_topic, pose_topic, goal_topic, rate_hz):
        super().__init__("waypoints_occ_visualizer")
        self.rate_hz = rate_hz
        self.lock = threading.Lock()

        # Latest data: occ as (grid_np, origin_x, origin_y, resolution)
        self.occ_data = None  # (np.ndarray (H,W), x_min, y_min, res) in map frame
        self.path_poses = []
        self.waypoints_poses = []
        self.robot_pose = None
        self.goal_pose = None

        self.sub_occ = self.create_subscription(
            OccupancyGrid, occ_topic, self._cb_occ, 10
        )
        self.sub_path = self.create_subscription(
            Path, path_topic, self._cb_path, 10
        )
        self.sub_waypoints = self.create_subscription(
            Path, waypoints_topic, self._cb_waypoints, 10
        )
        if pose_topic:
            self.sub_pose = self.create_subscription(
                PoseWithCovarianceStamped, pose_topic, self._cb_pose, 10
            )
        else:
            self.sub_pose = None

        if goal_topic:
            self.sub_goal = self.create_subscription(
                PoseStamped, goal_topic, self._cb_goal, 10
            )
        else:
            self.sub_goal = None

    def _cb_occ(self, msg):
        with self.lock:
            w, h = msg.info.width, msg.info.height
            res = msg.info.resolution
            ox = msg.info.origin.position.x
            oy = msg.info.origin.position.y
            arr = np.array(msg.data, dtype=np.int32).reshape((h, w))
            # OccupancyGrid: 0=free, 100=occupied, -1=unknown
            self.occ_data = (arr, ox, oy, res)

    def _cb_path(self, msg):
        with self.lock:
            self.path_poses = [
                (p.pose.position.x, p.pose.position.y) for p in msg.poses
            ]

    def _cb_waypoints(self, msg):
        with self.lock:
            self.waypoints_poses = [
                (p.pose.position.x, p.pose.position.y) for p in msg.poses
            ]

    def _cb_pose(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.robot_pose = (p.x, p.y)

    def _cb_goal(self, msg):
        with self.lock:
            p = msg.pose.position
            self.goal_pose = (p.x, p.y)

    def get_snapshot(self):
        with self.lock:
            occ = self.occ_data
            path = list(self.path_poses)
            waypoints = list(self.waypoints_poses)
            robot = self.robot_pose
            goal = self.goal_pose
        return occ, path, waypoints, robot, goal


def run_ros_spin(node):
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize waypoints and occupancy grid in real time"
    )
    parser.add_argument(
        "--occ-topic", default="/occupancy_grid",
        help="Occupancy grid (nav_msgs/OccupancyGrid)",
    )
    parser.add_argument("--path-topic", default="/planned_path", help="Planned path (nav_msgs/Path)")
    parser.add_argument("--waypoints-topic", default="/waypoints", help="Waypoints (nav_msgs/Path)")
    parser.add_argument(
        "--pose-topic", default="/pcl_pose",
        help="Robot pose (geometry_msgs/PoseWithCovarianceStamped)",
    )
    parser.add_argument(
        "--goal-topic", default="/move_base_simple/goal",
        help="Goal pose (geometry_msgs/PoseStamped)",
    )
    parser.add_argument("--rate", type=float, default=10.0, help="Plot update rate (Hz)")
    parser.add_argument("--view-col-min", type=int, default=-1, help="Min grid column to view")
    parser.add_argument("--view-col-max", type=int, default=-1, help="Max grid column to view")
    parser.add_argument("--view-row-min", type=int, default=-1, help="Min grid row to view")
    parser.add_argument("--view-row-max", type=int, default=-1, help="Max grid row to view")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = WaypointsOccVisualizer(
        args.occ_topic,
        args.path_topic,
        args.waypoints_topic,
        args.pose_topic or None,
        args.goal_topic or None,
        args.rate,
    )

    spin_thread = threading.Thread(target=run_ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Occupancy grid + waypoints (map frame)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    im_occ = None
    line_path, = ax.plot([], [], "b-", linewidth=2, label="Path")
    line_waypoints, = ax.plot([], [], "ro", markersize=6, label="Waypoints")
    robot_marker, = ax.plot([], [], "go", markersize=12, label="Robot")
    goal_marker, = ax.plot([], [], "y*", markersize=14, label="Goal")

    im_occ = None
    coord_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    last_mouse = {"x": None, "y": None}

    def on_move(event):
        if event.inaxes != ax:
            last_mouse["x"] = None
            last_mouse["y"] = None
            return
        last_mouse["x"] = event.xdata
        last_mouse["y"] = event.ydata

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    def update_with_occ(_frame):
        occ_data, path, waypoints, robot, goal = node.get_snapshot()
        nonlocal im_occ

        if occ_data is not None:
            arr, ox, oy, res = occ_data
            h, w = arr.shape
            extent = [ox, ox + w * res, oy, oy + h * res]
            vis = np.where(arr == -1, 127, 100 - arr)
            vis = np.clip(vis, 0, 100).astype(np.uint8)
            # OccupancyGrid row 0 = max y (origin.y + (height-1)*res); our bridge matches
            if im_occ is None:
                im_occ = ax.imshow(
                    vis,
                    cmap="gray",
                    extent=extent,
                    origin="upper",
                    alpha=0.8,
                    vmin=0,
                    vmax=100,
                )
            else:
                im_occ.set_data(vis)
                im_occ.set_extent(extent)

        if path:
            xs, ys = zip(*path)
            line_path.set_data(xs, ys)
            line_path.set_visible(True)
        else:
            line_path.set_data([], [])
            line_path.set_visible(False)

        if waypoints:
            xs, ys = zip(*waypoints)
            line_waypoints.set_data(xs, ys)
            line_waypoints.set_visible(True)
        else:
            line_waypoints.set_data([], [])
            line_waypoints.set_visible(True)

        if robot is not None:
            robot_marker.set_data([robot[0]], [robot[1]])
            robot_marker.set_visible(True)
        else:
            robot_marker.set_visible(False)

        if goal is not None:
            goal_marker.set_data([goal[0]], [goal[1]])
            goal_marker.set_visible(True)
        else:
            goal_marker.set_visible(False)

        all_x, all_y = [], []
        if path:
            all_x.extend([p[0] for p in path])
            all_y.extend([p[1] for p in path])
        if waypoints:
            all_x.extend([p[0] for p in waypoints])
            all_y.extend([p[1] for p in waypoints])
        if robot:
            all_x.append(robot[0])
            all_y.append(robot[1])
        if goal:
            all_x.append(goal[0])
            all_y.append(goal[1])
        if occ_data is not None:
            arr, ox, oy, res = occ_data
            h, w = arr.shape
            all_x.extend([ox, ox + w * res])
            all_y.extend([oy, oy + h * res])

        if occ_data is not None and (
            args.view_col_min >= 0
            or args.view_col_max >= 0
            or args.view_row_min >= 0
            or args.view_row_max >= 0
        ):
            col_min = args.view_col_min if args.view_col_min >= 0 else 0
            col_max = args.view_col_max if args.view_col_max >= 0 else w - 1
            row_min = args.view_row_min if args.view_row_min >= 0 else 0
            row_max = args.view_row_max if args.view_row_max >= 0 else h - 1

            col_min = max(0, min(col_min, w - 1))
            col_max = max(0, min(col_max, w - 1))
            row_min = max(0, min(row_min, h - 1))
            row_max = max(0, min(row_max, h - 1))

            x_min = ox + col_min * res
            x_max = ox + (col_max + 1) * res
            y_max = oy + (h - row_min) * res
            y_min = oy + (h - 1 - row_max) * res

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        elif all_x and all_y:
            margin = 0.5
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        if occ_data is not None and last_mouse["x"] is not None and last_mouse["y"] is not None:
            arr, ox, oy, res = occ_data
            h, w = arr.shape
            col = int((last_mouse["x"] - ox) / res)
            row = h - 1 - int((last_mouse["y"] - oy) / res)
            if 0 <= col < w and 0 <= row < h:
                coord_text.set_text(
                    f"col,row: {col}, {row}\n"
                    f"x,y: {last_mouse['x']:.2f}, {last_mouse['y']:.2f}"
                )
            else:
                coord_text.set_text("col,row: out of bounds")
        else:
            coord_text.set_text("")

        return line_path, line_waypoints, robot_marker, goal_marker, coord_text

    _ = animation.FuncAnimation(
        fig, update_with_occ, interval=1000.0 / args.rate, blit=False, cache_frame_data=False
    )
    ax.legend(loc="upper right")
    plt.tight_layout()

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    spin_thread.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

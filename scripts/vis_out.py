#!/usr/bin/env python3
"""Visualize predicted occupancy grids rotated into map orientation (OpenCV)."""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from path_planning.msg import AgentCenteredInput
from geometry_msgs.msg import PoseWithCovarianceStamped
import math

# Grid and layout constants
H = W = 201
NCOLS = 5
NROWS = 4
SCALE = 1          # upscale each cell for readability (201*2 = 402 px per cell)
PAD = 4            # pixels between cells
LABEL_H = 20       # pixels reserved for title text above each cell
CELL = H * SCALE
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1
BG = 40            # dark-grey background value


def _make_canvas():
    """Allocate the composite BGR canvas once."""
    cw = NCOLS * CELL + (NCOLS - 1) * PAD
    ch = NROWS * (CELL + LABEL_H) + (NROWS - 1) * PAD
    canvas = np.full((ch, cw, 3), BG, dtype=np.uint8)
    return canvas


def _cell_origin(row, col):
    """Top-left (x, y) of the image area for grid[row][col]."""
    x = col * (CELL + PAD)
    y = row * (CELL + LABEL_H + PAD) + LABEL_H
    return x, y


def _grid_to_bgr(grid_f32):
    """Convert 201×201 float32 [0,1] grid → upscaled BGR (white=free, black=occ)."""
    grey = (255 - (grid_f32 * 255).clip(0, 255)).astype(np.uint8)
    grey = cv2.resize(grey, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


class PredictionVisualizer(Node):
    # Row labels
    _ROW_LABELS = [
        [f'Input t-{4-t}' if t < 4 else 'Input t' for t in range(5)],
        [f'Pred t+{t+1}' for t in range(5)],
        [f'Binary t+{t+1}' for t in range(5)],
        ['', '', 'Overlay (all binary)', '', ''],
    ]

    def __init__(self):
        super().__init__('prediction_visualizer')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.sub = self.create_subscription(
            AgentCenteredInput,
            '/map_updater/predicted_grid_output',
            self._cb,
            qos,
        )
        self.sub_input = self.create_subscription(
            AgentCenteredInput,
            '/map_updater/occ_grid_input',
            self._input_cb,
            qos,
        )
        self.sub_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pcl_pose',
            self._pose_cb,
            10,
        )
        self.get_logger().info(
            'Subscribed to /map_updater/predicted_grid_output, '
            '/map_updater/occ_grid_input, and /pcl_pose')

        self.yaw = 0.0
        self.msg_count = 0
        self.input_mode = 1  # 0 = map_frame (no rotation), 1 = agent_centered (rotate by yaw)

        # Pre-allocate storage grids (float32, 201×201)
        self.grids_input = [np.zeros((H, W), np.float32) for _ in range(5)]
        self.grids_pred = [np.zeros((H, W), np.float32) for _ in range(5)]
        self.grids_binary = [np.zeros((H, W), np.float32) for _ in range(5)]
        self.grid_overlay = np.zeros((H, W), np.float32)

        # Canvas
        self.canvas = _make_canvas()
        self._draw_labels()

        cv2.namedWindow('Predictions', cv2.WINDOW_NORMAL)
        cv2.imshow('Predictions', self.canvas)
        cv2.waitKey(1)

    def _draw_labels(self):
        """Burn static title text onto the canvas."""
        for r in range(NROWS):
            for c in range(NCOLS):
                label = self._ROW_LABELS[r][c]
                if not label:
                    continue
                x, y = _cell_origin(r, c)
                # Put text just above the image area
                cv2.putText(self.canvas, label,
                            (x + 4, y - 5),
                            FONT, FONT_SCALE, (200, 200, 200), FONT_THICK,
                            cv2.LINE_AA)

    def _blit(self, row, col, grid_f32):
        """Write one upscaled grid into the canvas at (row, col)."""
        bgr = _grid_to_bgr(grid_f32)
        x, y = _cell_origin(row, col)
        self.canvas[y:y + CELL, x:x + CELL] = bgr

    def _show(self):
        cv2.imshow('Predictions', self.canvas)
        cv2.waitKey(1)

    # ---- ROS callbacks ----

    def _pose_cb(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _rotate_ego_to_map(self, grid):
        center = (100, 100)
        angle_deg = np.degrees(self.yaw)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(grid, M, (W, H),
                              flags=cv2.INTER_NEAREST,
                              borderValue=0.0)

    def _input_cb(self, msg):
        self.input_mode = msg.mode
        occ_fields = [msg.occ_0, msg.occ_1, msg.occ_2, msg.occ_3, msg.occ_4]
        for t in range(5):
            grid = np.array(occ_fields[t], dtype=np.float32)
            if grid.size != H * W:
                self.get_logger().warn(
                    f'Input occ_{t} has {grid.size} elements, expected {H*W}')
                continue
            grid = grid.reshape(H, W)
            if msg.mode == 1:
                grid = self._rotate_ego_to_map(grid)
            self.grids_input[t] = grid
            self._blit(0, t, grid)
        self._show()

    def _cb(self, msg):
        occ_fields = [msg.occ_0, msg.occ_1, msg.occ_2, msg.occ_3, msg.occ_4]
        for t in range(5):
            grid = np.array(occ_fields[t], dtype=np.float32)
            if grid.size != H * W:
                self.get_logger().warn(
                    f'occ_{t} has {grid.size} elements, expected {H*W}')
                continue
            grid = grid.reshape(H, W)
            if self.input_mode == 1:
                grid = self._rotate_ego_to_map(grid)
            self.grids_pred[t] = grid
            self._blit(1, t, grid)

            binary = np.where(grid < 0.5, 0.0, 1.0).astype(np.float32)
            self.grids_binary[t] = binary
            self._blit(2, t, binary)

        # Overlay (union of all binary)
        overlay = np.maximum.reduce(self.grids_binary)
        self.grid_overlay = overlay
        self._blit(3, 2, overlay)

        self.msg_count += 1
        self._show()


def main():
    rclpy.init()
    node = PredictionVisualizer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            # Allow OpenCV to process window events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print(f'\nStopped after {node.msg_count} messages.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
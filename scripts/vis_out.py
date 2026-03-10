#!/usr/bin/env python3
"""Visualize predicted occupancy grids rotated into map orientation."""

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from path_planning.msg import AgentCenteredInput
from geometry_msgs.msg import PoseWithCovarianceStamped
import math


class PredictionVisualizer(Node):
    def __init__(self):
        super().__init__('prediction_visualizer')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
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
            'Subscribed to /map_updater/predicted_grid_output, /map_updater/occ_grid_input, and /pcl_pose')

        self.yaw = 0.0

        # Set up 4x5 plot (4 rows, 5 columns)
        self.fig, self.axes = plt.subplots(4, 5, figsize=(20, 16))
        self.ims_input = []
        self.ims = []
        self.ims_thresh = []
        self.ims_overlay = []
        
        # Row 0: Input occupancy grids
        for t in range(5):
            im = self.axes[0, t].imshow(
                np.zeros((201, 201)),
                vmin=0.0, vmax=1.0,
                cmap='Greys',
                origin='lower',
            )
            time_label = f't-{4-t}' if t < 4 else 't'
            self.axes[0, t].set_title(f'Input {time_label}')
            self.axes[0, t].axis('off')
            self.ims_input.append(im)
        
        # Row 1: Original predictions (continuous values)
        for t in range(5):
            im = self.axes[1, t].imshow(
                np.zeros((201, 201)),
                vmin=0.0, vmax=1.0,
                cmap='Greys',
                origin='lower',
            )
            self.axes[1, t].set_title(f'Pred t+{t+1}')
            self.axes[1, t].axis('off')
            self.ims.append(im)
        
        # Row 2: Thresholded (binary)
        for t in range(5):
            im = self.axes[2, t].imshow(
                np.zeros((201, 201)),
                vmin=0.0, vmax=1.0,
                cmap='Greys',
                origin='lower',
            )
            self.axes[2, t].set_title(f'Binary t+{t+1}')
            self.axes[2, t].axis('off')
            self.ims_thresh.append(im)
        
        # Row 3: Overlay of all 5 thresholded frames (single frame, centered)
        # Hide columns 0, 1, 3, 4 and show overlay only in column 2
        for t in range(5):
            if t == 2:
                im = self.axes[3, t].imshow(
                    np.zeros((201, 201)),
                    vmin=0.0, vmax=1.0,
                    cmap='Greys',
                    origin='lower',
                )
                self.axes[3, t].set_title('Overlay (all binary)')
                self.axes[3, t].axis('off')
                self.ims_overlay.append(im)
            else:
                self.axes[3, t].axis('off')
            
        # self.fig.suptitle('Waiting for predictions…')
        self.fig.tight_layout()
        plt.ion()
        plt.show()

        self.msg_count = 0

    def _pose_cb(self, msg):
        q = msg.pose.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _rotate_ego_to_map(self, grid):
        """Rotate 201x201 ego grid by robot yaw so it aligns with map frame."""
        center = (100, 100)
        angle_deg = np.degrees(self.yaw)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(grid, M, (201, 201),
                                 flags=cv2.INTER_NEAREST,
                                 borderValue=0.0)
        return rotated

    def _threshold_binary(self, grid):
        """Threshold grid: < 0.5 -> 0 (white), >= 0.5 -> 1 (black)."""
        binary = np.where(grid < 0.5, 0.0, 1.0)
        return binary.astype(np.float32)

    def _overlay_frames(self, grids):
        """Overlay multiple frames: take max (union of obstacles)."""
        if not grids:
            return np.zeros((201, 201), dtype=np.float32)
        overlay = np.maximum.reduce(grids)
        return overlay.astype(np.float32)

    def _input_cb(self, msg):
        """Callback for input occupancy grids - same processing as predictions."""
        H = W = 201
        occ_fields = [msg.occ_0, msg.occ_1, msg.occ_2, msg.occ_3, msg.occ_4]

        for t in range(5):
            grid = np.array(occ_fields[t], dtype=np.float32)
            if grid.size == H * W:
                grid = grid.reshape(H, W)
            else:
                self.get_logger().warn(
                    f'Input occ_{t} has {grid.size} elements, expected {H*W}')
                continue
            
            # Rotate to map orientation (same as predictions)
            grid_rotated = self._rotate_ego_to_map(grid)
            
            # Display input in row 0 (same display logic as predictions)
            self.ims_input[t].set_data(grid_rotated)

    def _cb(self, msg):
        H = W = 201
        occ_fields = [msg.occ_0, msg.occ_1, msg.occ_2, msg.occ_3, msg.occ_4]

        binary_grids = []
        
        for t in range(5):
            grid = np.array(occ_fields[t], dtype=np.float32)
            if grid.size == H * W:
                grid = grid.reshape(H, W)
            else:
                self.get_logger().warn(
                    f'occ_{t} has {grid.size} elements, expected {H*W}')
                continue
            
            # Rotate to map orientation
            grid_rotated = self._rotate_ego_to_map(grid)
            
            # Display original in row 1
            self.ims[t].set_data(grid_rotated)
            
            # Threshold and display binary in row 2
            grid_binary = self._threshold_binary(grid_rotated)
            self.ims_thresh[t].set_data(grid_binary)
            binary_grids.append(grid_binary)
        
        # Create overlay of all 5 thresholded binary frames and display in row 3
        if len(binary_grids) == 5:
            overlay = self._overlay_frames(binary_grids)
            if self.ims_overlay:
                self.ims_overlay[0].set_data(overlay)

        self.msg_count += 1
        # self.fig.suptitle(f'Prediction #{self.msg_count}  yaw={np.degrees(self.yaw):.1f}°')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = PredictionVisualizer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            plt.pause(0.01)
    except KeyboardInterrupt:
        print(f'\nStopped after {node.msg_count} messages.')
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
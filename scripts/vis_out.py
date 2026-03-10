#!/usr/bin/env python3
"""Visualize predicted occupancy grids from /map_updater/predicted_grid_output."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from path_planning.msg import AgentCenteredInput


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
        self.get_logger().info(
            'Subscribed to /map_updater/predicted_grid_output')

        # Set up 1×5 plot
        self.fig, self.axes = plt.subplots(1, 5, figsize=(20, 4))
        self.ims = []
        for t in range(5):
            im = self.axes[t].imshow(
                np.zeros((201, 201)),
                vmin=0.0, vmax=1.0,
                cmap='Greys',
                origin='lower',
            )
            self.axes[t].set_title(f'Pred t+{t+1}')
            self.axes[t].axis('off')
            self.ims.append(im)
        self.fig.suptitle('Waiting for predictions…')
        self.fig.tight_layout()
        plt.ion()
        plt.show()

        self.msg_count = 0

    def _cb(self, msg):
        H = W = 201
        occ_fields = [msg.occ_0, msg.occ_1, msg.occ_2, msg.occ_3, msg.occ_4]

        for t in range(5):
            grid = np.array(occ_fields[t], dtype=np.float32)
            if grid.size == H * W:
                grid = grid.reshape(H, W)
            else:
                self.get_logger().warn(
                    f'occ_{t} has {grid.size} elements, expected {H*W}')
                continue
            self.ims[t].set_data(grid)

        self.msg_count += 1
        self.fig.suptitle(f'Prediction #{self.msg_count}')
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

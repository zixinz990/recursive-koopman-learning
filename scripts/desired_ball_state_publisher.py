from math import sqrt, floor, ceil
import numpy as np
import rclpy
from rclpy.node import Node
from platform_interfaces.msg import BallOdometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import matplotlib.pyplot as plt
import random
import pandas as pd

def uniform_pos_in_platform():
    pos_list = []
    r_max = 0.19
    x = -r_max
    while x <= r_max:
        y = -r_max
        while y <= r_max:
            # If the sample is inside the platform
            if np.abs(y) + np.abs(x) / np.sqrt(3.) < r_max:
                pos_list.append(np.array([x, y]))
            y += 0.02
        x += 0.02
    return pos_list

def hexagon_grid_vertices(S, s, r_max=0.19):
    vertices = []
    delta_y = s * sqrt(3) / 2  # Vertical distance between rows
    y_max = S * sqrt(3) / 2  # Maximum y-coordinate
    j_max = int(ceil(y_max / delta_y))  # Maximum row index

    for j in range(-j_max, j_max + 1):
        y = delta_y * j
        y_abs = abs(y)
        x_limit = S - (2 / sqrt(3)) * y_abs

        if x_limit < 0:
            continue  # Skip rows outside the hexagon

        if j % 2 == 0:
            # Even row
            imin = int(ceil((-x_limit) / s))
            imax = int(floor(x_limit / s))
            for i in range(imin, imax + 1):
                x = s * i
                if (np.abs(y) + np.abs(x) / np.sqrt(3.) < r_max) and (np.abs(x) < r_max):
                    vertices.append(np.array([x, y]))
        else:
            # Odd row
            imin = int(ceil((-x_limit) / s - 0.5))
            imax = int(floor(x_limit / s - 0.5))
            for i in range(imin, imax + 1):
                x = s * (i + 0.5)
                if (np.abs(y) + np.abs(x) / np.sqrt(3.) < r_max) and (np.abs(x) < r_max):
                    vertices.append(np.array([x, y]))
    
    # Randomly shuffle the vertices
    random.shuffle(vertices)

    # Add a useless vertex at the beginning
    vertices.insert(0, np.array([0.005, 0.005]))

    return vertices

class BallPosePublisher(Node):
    def __init__(self, trial_time):
        super().__init__('desired_ball_state_pub')
        self.trial_time = trial_time
        self.publisher_ = self.create_publisher(BallOdometry, 'desired_ball_state', 2)
        self.timer = self.create_timer(1.0 / 100.0, self.publish_ball_pose)  # 100 Hz

        self.target_positions = hexagon_grid_vertices(0.3, 0.02)
        print("Number of target positions: " + str(len(self.target_positions)))
        self.time_counter = 0
        self.pos_counter = 0
        self.target_pos = self.target_positions[0]

        self.get_logger().info("desired_ball_state_pub is publishing to /desired_ball_state at 100 Hz")
        print("Pos counter: " + str(self.pos_counter) + ", desired pos in cm: " + str(self.target_pos * 100.0))

    def publish_ball_pose(self):
        if self.time_counter > self.trial_time * 100:
            self.time_counter = 0 # reset time_counter
            self.pos_counter += 1
            self.target_pos = self.target_positions[self.pos_counter]
            print("Pos counter: " + str(self.pos_counter) + ", desired pos in cm: " + str(self.target_pos * 100.0))

        # Publish
        ball_pose = BallOdometry()
        ball_pose.x = self.target_pos[0]
        ball_pose.y = self.target_pos[1]
        ball_pose.z = 0.0
        ball_pose.xdot = 0.0
        ball_pose.ydot = 0.0
        ball_pose.zdot = 0.0
        self.publisher_.publish(ball_pose)

        # Update time_counter
        self.time_counter = self.time_counter + 1

def main(args=None):
    rclpy.init(args=args)
    trial_time = int(input('Enter the time length for each trial (in sec): '))
    ball_pose_publisher = BallPosePublisher(trial_time)

    try:
        rclpy.spin(ball_pose_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        ball_pose_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    # pos_arr = hexagon_grid_vertices(0.5, 0.02, r_max = 0.25)
    # plt.scatter([pos[0] for pos in pos_arr], [pos[1] for pos in pos_arr])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

from math import sqrt, floor, ceil
import numpy as np
import rclpy
from rclpy.node import Node
from platform_interfaces.msg import BallOdometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import matplotlib.pyplot as plt
import random
import pandas as pd


def linear_state_traj(p1, p2, T, dt):
    if T <= 0:
        raise ValueError("Total time T must be greater than zero.")
    if dt <= 0:
        raise ValueError("Time step dt must be greater than zero.")
    if T < dt:
        raise ValueError("Total time T must be greater than or equal to time step dt.")    
    
    vel = (p2 - p1) / T
    time_steps = np.arange(0, T + dt, dt)
    p_traj = []
    v_traj = []
    for t in time_steps:
        p_traj.append(p1 + vel * t)
        v_traj.append(vel)
    p_traj = np.array(p_traj)
    v_traj = np.array(v_traj)
    x_traj = np.hstack((p_traj, v_traj))    
    return x_traj[:-1]

def generate_ref_traj():
    # pos_1 = np.array([0.09, -0.18])
    # pos_2 = np.array([0.09, 0.18])
    # pos_3 = np.array([-0.18, 0.0])
    # traj_1 = linear_state_traj(pos_1, pos_3, 5.0, 0.01)
    # traj_12 = linear_state_traj(pos_3, pos_3, 5.0, 0.01)
    # traj_2 = linear_state_traj(pos_3, pos_2, 5.0, 0.01)
    # traj_23 = linear_state_traj(pos_2, pos_2, 5.0, 0.01)
    # traj_3 = linear_state_traj(pos_2, pos_1, 5.0, 0.01)
    # traj = np.vstack((traj_1, traj_12, traj_2, traj_23, traj_3))

    pos_1 = np.array([-0.09, -0.18])
    pos_2 = np.array([-0.09, 0.18])
    pos_3 = np.array([0.09, -0.18])
    pos_4 = np.array([0.09, 0.18])

    traj_1 = linear_state_traj(pos_1, pos_2, 7.0, 0.01)
    traj_12 = linear_state_traj(pos_2, pos_2, 2.0, 0.01)
    traj_2 = linear_state_traj(pos_2, pos_3, 7.0, 0.01)
    traj_23 = linear_state_traj(pos_3, pos_3, 2.0, 0.01)
    traj_3 = linear_state_traj(pos_3, pos_4, 7.0, 0.01)
    traj_34 = linear_state_traj(pos_4, pos_4, 10.0, 0.01)
    traj = np.vstack((traj_1, traj_12, traj_2, traj_23, traj_3, traj_34))

    return traj


class BallPosePublisher(Node):
    def __init__(self):
        super().__init__('desired_ball_state_pub')
        self.publisher_ = self.create_publisher(BallOdometry, 'desired_ball_state', 2)
        self.timer = self.create_timer(1.0 / 100.0, self.publish_ball_pose)  # 100 Hz

        self.target_positions = generate_ref_traj()
        print("Number of target positions: " + str(len(self.target_positions)))
        self.init_pos_pub_time = 20.0
        self.time_counter = 0
        self.pos_counter = 0
        self.target_pos = self.target_positions[0]

        self.get_logger().info("desired_ball_state_pub is publishing to /desired_ball_state at 100 Hz")
        print("Pos counter: " + str(self.pos_counter) + ", desired pos in cm: " + str(self.target_pos * 100.0))

    def publish_ball_pose(self):
        if self.time_counter * 0.01 < self.init_pos_pub_time:
            self.target_pos = np.hstack((self.target_positions[0, :2], np.array([0.0, 0.0])))
        else:
            self.pos_counter += 1
            if self.pos_counter > len(self.target_positions) - 1:
                self.pos_counter = 0
                self.time_counter = 0
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
        print("Time counter: " + str(self.time_counter))
        self.time_counter = self.time_counter + 1

def main(args=None):
    rclpy.init(args=args)
    ball_pose_publisher = BallPosePublisher()

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

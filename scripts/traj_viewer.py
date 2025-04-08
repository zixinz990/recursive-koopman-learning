#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sys
import matplotlib
# Ensure we are using the Qt5Agg backend
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from platform_interfaces.msg import BallOdometry

class CoordinatePlotter(Node):
    def __init__(self):
        super().__init__('coordinate_plotter')
        self.ball_pos_sub = self.create_subscription(
            BallOdometry, '/filtered_ball_pose', self.ball_pos_callback, 2)
        self.des_ball_pos_sub = self.create_subscription(
            BallOdometry, 'desired_ball_state', self.des_ball_pos_callback, 2)
        self.ball_pos_sub  # Prevent unused variable warning

        self.x_data = []
        self.y_data = []
        self.des_x = 0.0
        self.des_y = 0.0
        self.des_x_last = 0.0
        self.des_y_last = 0.0

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo', markersize=1, label='Trajectory')
        self.des_point, = self.ax.plot([], [], 'go', markersize=5, label='Desired Position')
        self.current_point, = self.ax.plot([], [], 'ro', markersize=3, label='Current Position')

        self.ax.set_xlim(-25, 25)
        self.ax.set_ylim(-25, 25)
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.ax.grid(True)  # Add grid to the plot

        # Start ROS2 spinning in a separate thread
        ros_thread = threading.Thread(target=self.run_ros_spin)
        ros_thread.daemon = True
        ros_thread.start()

        # Start the animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

        # Show the plot (this will block)
        plt.show()

    def run_ros_spin(self):
        rclpy.spin(self)

    def ball_pos_callback(self, msg):
        self.x_data.append(msg.x * 100.0)
        self.y_data.append(msg.y * 100.0)

    def des_ball_pos_callback(self, msg):
        self.des_x_last = self.des_x
        self.des_y_last = self.des_y

        self.des_x = msg.x * 100.0
        self.des_y = msg.y * 100.0

        # if self.des_x_last != self.des_x or self.des_y_last != self.des_y:
        #     self.x_data = []
        #     self.y_data = []

    def update_plot(self, frame):
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)

        # Update the current position point
        if self.x_data and self.y_data:
            self.current_point.set_xdata(self.x_data[-1])
            self.current_point.set_ydata(self.y_data[-1])

        # Update the desired position point
        if self.des_x and self.des_y:
            self.des_point.set_xdata(self.des_x)
            self.des_point.set_ydata(self.des_y)

        return self.line, self.current_point, self.des_point

def main(args=None):
    rclpy.init(args=args)
    coordinate_plotter = CoordinatePlotter()
    # No need to call rclpy.spin() here since it's handled in the class
    rclpy.shutdown()

if __name__ == '__main__':
    main()

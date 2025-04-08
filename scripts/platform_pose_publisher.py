from math import sqrt, floor, ceil
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from platform_interfaces.msg import BallOdometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import matplotlib.pyplot as plt
import random

class PlatformPosePublisher(Node):
    def __init__(self):
        super().__init__('platform_pose_publisher')
        self.publisher_ = self.create_publisher(Pose, 'spacemouse_pose', 100)
        self.timer = self.create_timer(0.01, self.publish_state)  # Adjust the publish rate as needed

    def publish_state(self):        
        # Publish
        ball_pose = Pose()
        ball_pose.position.x = -0.21142857142857144
        ball_pose.position.y = 0.03428571428571429
        ball_pose.position.z = 0.0
        ball_pose.orientation.x = -0.479425538604203
        ball_pose.orientation.y = 0.0
        ball_pose.orientation.z = 0.0
        ball_pose.orientation.w = 0.8775825618903728
        self.publisher_.publish(ball_pose)

def main(args=None):
    rclpy.init(args=args)
    platform_pose_publisher = PlatformPosePublisher()

    try:
        rclpy.spin(platform_pose_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        platform_pose_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    # pos_arr = np.array(hexagon_grid_vertices(0.3, 0.02))
    # print(np.shape(pos_arr))
    # plt.scatter(pos_arr[:, 0], pos_arr[:, 1])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

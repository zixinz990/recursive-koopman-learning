#pragma once

#include <mutex>

#include <rclcpp/rclcpp.hpp>

#include "platform_interfaces/msg/ball_odometry.hpp"
#include "platform_interfaces/msg/servo_angles.hpp"

#include "rkl_cpp/LQR.hpp"
#include "rkl_cpp/SAC.hpp"

using namespace std;

class Interface {
public:
    Interface(shared_ptr<rclcpp::Node> &node, shared_ptr<mutex> &mtx);

    bool ctrl_update();

    bool fbk_update();

    bool model_update();

    bool send_cmd();

    void declare_parameters();

    StewartState stewart_state;

private:
    // ROS
    shared_ptr<rclcpp::Node> node;
    rclcpp::Publisher<platform_interfaces::msg::ServoAngles>::SharedPtr pub_servo_angles;
    rclcpp::Subscription<platform_interfaces::msg::BallOdometry>::SharedPtr sub_ball_odom;
    rclcpp::Subscription<platform_interfaces::msg::BallOdometry>::SharedPtr sub_ball_state_d;
    platform_interfaces::msg::ServoAngles servo_angles_msg = platform_interfaces::msg::ServoAngles();

    // ROS callback functions
    void ball_state_callback(platform_interfaces::msg::BallOdometry::SharedPtr ball_odom);

    void ball_state_d_callback(platform_interfaces::msg::BallOdometry::SharedPtr ball_state_d);

    // Thread lock
    shared_ptr<mutex> mtx;

    // Control policy pointer
    unique_ptr<LQR> lqr_ptr;
    unique_ptr<SAC> sac_ptr;

    // Active learning related
    double learn_time; // in ms
    size_t learn_time_counter;
};

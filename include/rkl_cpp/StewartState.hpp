#pragma once

#include <memory>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"

#include "rkl_cpp/ConstParams.hpp"
#include "rkl_cpp/Koopman.hpp"
#include "rkl_cpp/Utils.hpp"

using namespace std;
using namespace Eigen;

class StewartFeedback {
public:
    StewartFeedback() { reset(); }

    void reset();

    Vector4d ball_state; // latest feedback
    VectorXd ball_obs; // current observations

    Vector4d ball_state_ils; // x1, "current" state, for model update through ILS
    VectorXd ball_obs_ils; // z1, observations of x1, for model update through ILS

    Vector4d ball_state_prev_ils; // x0, "previous" state, for model update through ILS
    VectorXd ball_obs_prev_ils; // z0, observations of x0, for model update through ILS

    FixedSizeQueue<VectorXd> ball_state_queue = FixedSizeQueue<VectorXd>(20);
    FixedSizeQueue<double> ball_time_queue = FixedSizeQueue<double>(20);
};

class StewartControl {
public:
    StewartControl() { reset(); }

    void reset();

    // Desired ball state
    Vector4d ball_state_d_prev;
    Vector4d ball_state_d;
    VectorXd ball_obs_d;

    // Servo commands
    VectorXd servo_cmd; // latest solution, sent to ROS
    VectorXd servo_cmd_d;  // reference
    VectorXd servo_cmd_ils; // u0, "current" input, for model update through ILS

    FixedSizeQueue<VectorXd> servo_cmd_queue = FixedSizeQueue<VectorXd>(20);
    FixedSizeQueue<double> servo_cmd_time_queue = FixedSizeQueue<double>(20);
};

class StewartParams {
public:
    StewartParams() { reset(); }

    void reset();

    void load(const shared_ptr<rclcpp::Node> &node);

    void print();

    size_t ctrl_type;
    size_t online_update;

    // MPC parameters
    int ctrl_horizon;
    VectorXd q_weights;
    VectorXd r_weights;
    VectorXd qf_weights;

    // Active learning parameters
    double learn_weight; // the weight of learning cost in the running cost
    VectorXd sac_r_weights; // bound the change of policy insertion

    // Servo commands bounds
    VectorXd input_lb;
    VectorXd input_ub;

    // Input moving average filter
    size_t input_maf_window_size;
};

class StewartModel {
public:
    Koopman koopman_model = Koopman(CONTROL_PERIOD / 1000.0); // Koopman model
};

class StewartState {
public:
    StewartState() {
        fbk.reset();
        ctrl.reset();
        params.reset();
    }

    StewartFeedback fbk;
    StewartControl ctrl;
    StewartParams params;
    StewartModel model;
};

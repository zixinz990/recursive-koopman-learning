#include "rkl_cpp/StewartState.hpp"

void StewartFeedback::reset() {
    ball_state.setZero();
    ball_obs.resize(NUM_STATE_OBS);
    ball_obs.setZero();

    ball_state_ils.setZero();
    ball_obs_ils.resize(NUM_STATE_OBS);
    ball_obs_ils.setZero();

    ball_state_prev_ils.setZero();
    ball_obs_prev_ils.resize(NUM_STATE_OBS);
    ball_obs_prev_ils.setZero();
}

void StewartControl::reset() {
    // Desired ball state
    ball_state_d_prev.setZero();
    ball_state_d.setZero();
    ball_obs_d.resize(NUM_STATE_OBS);
    ball_obs_d.setZero();

    // Servo commands
    servo_cmd.resize(6);
    servo_cmd.setZero();
    servo_cmd_d.resize(6);
    servo_cmd_d << 45.0, 45.0, 45.0, 45.0, 45.0, 45.0;
    servo_cmd_ils.resize(6);
    servo_cmd_ils.setZero();
}

void StewartParams::reset() {
    ctrl_type = 0; // 0: LQR; 1: SAC
    online_update = 0; // 0: disable; 1: enable

    // MPC parameters
    ctrl_horizon = 20;
    q_weights.resize(NUM_STATE_OBS);
    q_weights.setZero();
    r_weights.resize(NUM_INPUT_OBS);
    r_weights.setOnes();
    qf_weights.resize(NUM_STATE_OBS);
    qf_weights.setZero();

    // Active learning parameters
    learn_weight = 100.0;
    sac_r_weights.resize(NUM_INPUT_OBS);
    sac_r_weights.setOnes() * 0.003;

    input_lb.resize(NUM_INPUT_OBS);
    input_lb.setOnes();
    input_ub.resize(NUM_INPUT_OBS);
    input_ub.setOnes();

    // Input moving average filter
    input_maf_window_size = 5;
}

void StewartParams::load(const shared_ptr<rclcpp::Node> &node) {
    cout << "Loading parameters..." << endl;

    node->get_parameter("ctrl_type", ctrl_type);
    node->get_parameter("online_update", online_update);

    // MPC parameters
    node->get_parameter("ctrl_horizon", ctrl_horizon);

    double mpc_q_weights_x, mpc_q_weights_y, mpc_q_weights_vx, mpc_q_weights_vy, mpc_q_weights_obs;
    node->get_parameter("mpc_q_weights_x", mpc_q_weights_x);
    node->get_parameter("mpc_q_weights_y", mpc_q_weights_y);
    node->get_parameter("mpc_q_weights_vx", mpc_q_weights_vx);
    node->get_parameter("mpc_q_weights_vy", mpc_q_weights_vy);
    node->get_parameter("mpc_q_weights_obs", mpc_q_weights_obs);
    q_weights.head(4)
            << mpc_q_weights_x, mpc_q_weights_y, mpc_q_weights_vx, mpc_q_weights_vy;
    q_weights.tail(NUM_STATE_OBS - 4).setConstant(mpc_q_weights_obs);

    double mpc_r_weights_1, mpc_r_weights_2, mpc_r_weights_3, mpc_r_weights_4, mpc_r_weights_5, mpc_r_weights_6;
    node->get_parameter("mpc_r_weights_1", mpc_r_weights_1);
    node->get_parameter("mpc_r_weights_2", mpc_r_weights_2);
    node->get_parameter("mpc_r_weights_3", mpc_r_weights_3);
    node->get_parameter("mpc_r_weights_4", mpc_r_weights_4);
    node->get_parameter("mpc_r_weights_5", mpc_r_weights_5);
    node->get_parameter("mpc_r_weights_6", mpc_r_weights_6);
    r_weights << mpc_r_weights_1, mpc_r_weights_2, mpc_r_weights_3,
            mpc_r_weights_4, mpc_r_weights_5, mpc_r_weights_6;

    double mpc_qf_weights_x, mpc_qf_weights_y, mpc_qf_weights_vx, mpc_qf_weights_vy, mpc_qf_weights_obs;
    node->get_parameter("mpc_qf_weights_x", mpc_qf_weights_x);
    node->get_parameter("mpc_qf_weights_y", mpc_qf_weights_y);
    node->get_parameter("mpc_qf_weights_vx", mpc_qf_weights_vx);
    node->get_parameter("mpc_qf_weights_vy", mpc_qf_weights_vy);
    node->get_parameter("mpc_qf_weights_obs", mpc_qf_weights_obs);
    qf_weights.head(4)
            << mpc_qf_weights_x, mpc_qf_weights_y, mpc_qf_weights_vx, mpc_qf_weights_vy;
    qf_weights.tail(NUM_STATE_OBS - 4).setConstant(mpc_qf_weights_obs);

    // Active learning parameters
    node->get_parameter("learn_weight", learn_weight);

    double sac_r_weights_;
    node->get_parameter("sac_r_weights", sac_r_weights_);
    sac_r_weights.resize(NUM_INPUT_OBS);
    sac_r_weights.setOnes();
    sac_r_weights = sac_r_weights_ * sac_r_weights;

    // Servo commands bounds
    double input_lb_, input_ub_;
    node->get_parameter("input_lb", input_lb_);
    node->get_parameter("input_ub", input_ub_);
    input_lb = input_lb_ * input_lb;
    input_ub = input_ub_ * input_ub;

    // Input moving average filter
    node->get_parameter("input_maf_window_size", input_maf_window_size);

    print();
}

void StewartParams::print() {
    if (ctrl_type == 0) cout << "ctrl_type: " << "LQR" << endl;
    if (ctrl_type == 1) cout << "ctrl_type: " << "SAC" << endl;
    if (online_update == 0) cout << "Online update is disabled" << endl;
    if (online_update == 1) cout << "Online update is enabled" << endl;
    cout << "ctrl_horizon: " << ctrl_horizon << endl;
    cout << "q_weights: " << q_weights.transpose() << endl;
    cout << "r_weights: " << r_weights.transpose() << endl;
    cout << "qf_weights: " << qf_weights.transpose() << endl;
    cout << "learn_weight: " << learn_weight << endl;
    cout << "sac_r_weights: " << sac_r_weights.transpose() << endl;
    cout << "input_lb: " << input_lb.transpose() << endl;
    cout << "input_ub: " << input_ub.transpose() << endl << endl;
}

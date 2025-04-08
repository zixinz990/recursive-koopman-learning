#include "rkl_cpp/Interface.hpp"

Interface::Interface(shared_ptr<rclcpp::Node> &node_, shared_ptr<mutex> &mtx_) {
    // ROS
    node = node_;
    pub_servo_angles = node_->create_publisher<platform_interfaces::msg::ServoAngles>(
            "/servo_angles", 1);
    sub_ball_odom = node_->create_subscription<platform_interfaces::msg::BallOdometry>(
            "/filtered_ball_pose", 1,
            bind(&Interface::ball_state_callback, this, std::placeholders::_1)
    );
    sub_ball_state_d = node_->create_subscription<platform_interfaces::msg::BallOdometry>(
            "/desired_ball_state", 1,
            bind(&Interface::ball_state_d_callback, this, std::placeholders::_1)
    );

    // Initialize parameters
    declare_parameters(); // declare ROS parameters
    stewart_state.params.load(node_); // load from ROS

    // Calculate the observations of the desired state
    stewart_state.ctrl.ball_obs_d = Utils::poly_obs(stewart_state.ctrl.ball_state_d);

    // Initialize policies
    lqr_ptr = make_unique<LQR>();
    lqr_ptr->init(stewart_state);
    sac_ptr = make_unique<SAC>();
    sac_ptr->init(stewart_state);

    // Thread lock
    mtx = mtx_;

    // Active learning related
    learn_time = 12.5 * 1000.0; // in ms
    learn_time_counter = 0;
}

bool Interface::ctrl_update() {
    double start = chrono::duration<double, milli>(
            chrono::system_clock::now().time_since_epoch()).count();
    mtx->lock();

    // Solve OCP
    VectorXd u_star;
    if (stewart_state.params.ctrl_type == 0) {
        lqr_ptr->cal_action(stewart_state);
        u_star = lqr_ptr->optimal_action;
        // cout << "LQR solution: " << u_star.transpose() << endl;
        if (!stewart_state.ctrl.ball_state_d.isApprox(stewart_state.ctrl.ball_state_d_prev)) {
            // Reset the model after each trial
            stewart_state.model.koopman_model.reset(CONTROL_PERIOD / 1000.0);
            cout << "The Koopman model is reset!" << endl;
        }
    } else if (stewart_state.params.ctrl_type == 1) {
        sac_ptr->cal_action(stewart_state);
        u_star = sac_ptr->optimal_action;
        // cout << "SAC solution: " << u_star.transpose() << endl;
        if (stewart_state.ctrl.ball_state_d.isApprox(stewart_state.ctrl.ball_state_d_prev)) {
            // Learn a while
            learn_time_counter += 1;
            sac_ptr->task.learn_weight = stewart_state.params.learn_weight;
            if (learn_time_counter >= learn_time / CONTROL_PERIOD) {
                sac_ptr->task.learn_weight = 0.0;
            }
        } else {
            // Reset the model and learning weight after each trial
            // stewart_state.model.koopman_model.reset(CONTROL_PERIOD / 1000.0);
            // cout << "The Koopman model is reset!" << endl;
            // learn_time_counter = 0;
            // sac_ptr->task.learn_weight = stewart_state.params.learn_weight;
        }
        // cout << "Learn weight: " << sac_ptr->task.learn_weight << endl;
    }
    stewart_state.ctrl.servo_cmd = u_star.cwiseMax(stewart_state.params.input_lb).cwiseMin(
            stewart_state.params.input_ub);
    cout << "Servo command: " << stewart_state.ctrl.servo_cmd.transpose() << endl;

    // Update target history
    stewart_state.ctrl.ball_state_d_prev = stewart_state.ctrl.ball_state_d;

    mtx->unlock();
    double end = chrono::duration<double, milli>(
            chrono::system_clock::now().time_since_epoch()).count();
    cout << "The control thread takes " << end - start << " ms to update" << endl;
    return true;
}

bool Interface::fbk_update() {
    if (stewart_state.fbk.ball_state_queue.is_full() &&
        stewart_state.ctrl.servo_cmd_queue.is_full()) {
        // Get interpolation timestamps
        vector<double> timestamp_interp;
        timestamp_interp.resize(2);

        mtx->lock();
        if (stewart_state.fbk.ball_time_queue.vec.back() >
            stewart_state.ctrl.servo_cmd_time_queue.vec.back()) {
            timestamp_interp[1] = stewart_state.ctrl.servo_cmd_time_queue.vec.back(); // in ms, timestamp for x1
        } else {
            timestamp_interp[1] = stewart_state.fbk.ball_time_queue.vec.back(); // in ms, timestamp for x1
        }
        timestamp_interp[0] = timestamp_interp[1] - CONTROL_PERIOD; // timestamp for x0
        try {
            // Interpolate ball state trajectory
            vector<VectorXd> ball_state_interp = Utils::interp1d(
                    stewart_state.fbk.ball_time_queue.vec,
                    stewart_state.fbk.ball_state_queue.vec,
                    timestamp_interp);

            // Interpolate servo cmd trajectory
            vector<VectorXd> servo_cmd_interp = Utils::interp1d(
                    stewart_state.ctrl.servo_cmd_time_queue.vec,
                    stewart_state.ctrl.servo_cmd_queue.vec,
                    timestamp_interp);

            // Update x0 and x1
            stewart_state.fbk.ball_state_prev_ils = ball_state_interp[0];
            stewart_state.fbk.ball_state_ils = ball_state_interp[1];

            // Update u0
            stewart_state.ctrl.servo_cmd_ils = servo_cmd_interp[0];
        }
        catch (const exception &e) {
            cerr << "Interpolation error: " << e.what() << endl;
        }
        mtx->unlock();
    }
    return true;
}

bool Interface::model_update() {
    mtx->lock();
    // Vector2d tmp_vec;
    // tmp_vec << 9.0, -18.0;
    // Vector2d dist = tmp_vec - stewart_state.fbk.ball_state.head(2);
    // cout << "Distance: " << dist.norm() << endl;
    // if (dist.norm() > 4.2) {
    //     stewart_state.model.koopman_model.update_model(stewart_state.fbk.ball_state_prev_ils,
    //                                                    stewart_state.ctrl.servo_cmd_ils,
    //                                                    stewart_state.fbk.ball_state_ils);
    // }
    if (stewart_state.params.online_update == 1) {
        stewart_state.model.koopman_model.update_model(stewart_state.fbk.ball_state_prev_ils,
                                                       stewart_state.ctrl.servo_cmd_ils,
                                                       stewart_state.fbk.ball_state_ils);
        // stewart_state.model.koopman_model.edmd(stewart_state.fbk.ball_state_prev_ils,
        //                                        stewart_state.ctrl.servo_cmd_ils,
        //                                        stewart_state.fbk.ball_state_ils);
    }
    mtx->unlock();
    return true;
}

bool Interface::send_cmd() {
    mtx->lock();
    stewart_state.ctrl.servo_cmd_queue.push(stewart_state.ctrl.servo_cmd);
    if (stewart_state.ctrl.servo_cmd_queue.is_full()) {
        // Moving average filter
        vector<VectorXd> vec_tmp(stewart_state.ctrl.servo_cmd_queue.vec.end() -
                                 stewart_state.params.input_maf_window_size,
                                 stewart_state.ctrl.servo_cmd_queue.vec.end());
        VectorXd sum;
        sum.resize(6);
        sum.setZero();
        for (const auto &i: vec_tmp) sum += i;
        VectorXd servo_cmd_avg = sum / vec_tmp.size();

        // Publish
        servo_angles_msg.angle_1 = servo_cmd_avg[0];
        servo_angles_msg.angle_2 = servo_cmd_avg[1];
        servo_angles_msg.angle_3 = servo_cmd_avg[2];
        servo_angles_msg.angle_4 = servo_cmd_avg[3];
        servo_angles_msg.angle_5 = servo_cmd_avg[4];
        servo_angles_msg.angle_6 = servo_cmd_avg[5];
        pub_servo_angles->publish(servo_angles_msg);

        // Record the timestamp
        double timestamp = chrono::duration<double, milli>(
                chrono::system_clock::now().time_since_epoch()).count();
        stewart_state.ctrl.servo_cmd_time_queue.push(timestamp);
    }
    mtx->unlock();
    return true;
}

void Interface::declare_parameters() {
    node->declare_parameter<int>("ctrl_type", 0);
    node->declare_parameter<int>("online_update", 0);

    node->declare_parameter<int>("ctrl_horizon", 20);

    node->declare_parameter<double>("mpc_q_weights_x", 50.0);
    node->declare_parameter<double>("mpc_q_weights_y", 50.0);
    node->declare_parameter<double>("mpc_q_weights_vx", 1.0);
    node->declare_parameter<double>("mpc_q_weights_vy", 1.0);
    node->declare_parameter<double>("mpc_q_weights_obs", 0.0);

    node->declare_parameter<double>("mpc_r_weights_1", 0.6);
    node->declare_parameter<double>("mpc_r_weights_2", 0.6);
    node->declare_parameter<double>("mpc_r_weights_3", 0.6);
    node->declare_parameter<double>("mpc_r_weights_4", 0.6);
    node->declare_parameter<double>("mpc_r_weights_5", 0.6);
    node->declare_parameter<double>("mpc_r_weights_6", 0.6);

    node->declare_parameter<double>("mpc_qf_weights_x", 0.0);
    node->declare_parameter<double>("mpc_qf_weights_y", 0.0);
    node->declare_parameter<double>("mpc_qf_weights_vx", 0.0);
    node->declare_parameter<double>("mpc_qf_weights_vy", 0.0);
    node->declare_parameter<double>("mpc_qf_weights_obs", 0.0);

    node->declare_parameter<double>("learn_weight", 100.0);

    node->declare_parameter<double>("sac_r_weights", 0.003);

    node->declare_parameter<double>("input_lb", 45.0);
    node->declare_parameter<double>("input_ub", 100.0);

    node->declare_parameter<int>("input_maf_window_size", 5);
}

void
Interface::ball_state_callback(const platform_interfaces::msg::BallOdometry::SharedPtr ball_odom) {
    mtx->lock();
    stewart_state.fbk.ball_state << 100.0 * ball_odom->x, 100.0 * ball_odom->y,
            100.0 * ball_odom->xdot, 100.0 * ball_odom->ydot;
    stewart_state.fbk.ball_obs = Utils::poly_obs(stewart_state.fbk.ball_state);
    stewart_state.fbk.ball_state_queue.push(Vector4d(100.0 * ball_odom->x, 100.0 * ball_odom->y,
                                                     100.0 * ball_odom->xdot,
                                                     100.0 * ball_odom->ydot));

    // Record the timestamp
    double timestamp = chrono::duration<double, milli>(
            chrono::system_clock::now().time_since_epoch()).count();
    stewart_state.fbk.ball_time_queue.push(timestamp);
    mtx->unlock();
}

void
Interface::ball_state_d_callback(platform_interfaces::msg::BallOdometry::SharedPtr ball_state_d) {
    stewart_state.ctrl.ball_state_d << 100.0 * ball_state_d->x, 100.0 * ball_state_d->y,
            100.0 * ball_state_d->xdot, 100.0 * ball_state_d->ydot;
    stewart_state.ctrl.ball_obs_d = Utils::poly_obs(stewart_state.ctrl.ball_state_d);
}

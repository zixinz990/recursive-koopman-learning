#include "rkl_cpp/LQR.hpp"

void LQR::init(StewartState &stewart_state) {
    set_params(stewart_state);
    set_dynamics(stewart_state);
    set_target(stewart_state);
}

void LQR::update(StewartState &stewart_state) {
    set_dynamics(stewart_state);
    set_target(stewart_state);
}

void LQR::set_params(StewartState &stewart_state) {
    set_params(CONTROL_PERIOD / 1000.0, stewart_state.params.ctrl_horizon,
               stewart_state.params.input_lb, stewart_state.params.input_ub,
               stewart_state.params.q_weights, stewart_state.params.r_weights,
               stewart_state.params.qf_weights);
}

void LQR::set_params(const double dt_, const size_t horizon_,
                     const Eigen::VectorXd &input_lb_, const Eigen::VectorXd &input_ub_,
                     const Eigen::VectorXd &q_weights_, const Eigen::VectorXd &r_weights_,
                     const Eigen::VectorXd &qf_weights_) {
    // Parameters
    dt = dt_;
    horizon = horizon_;
    input_lb = input_lb_;
    input_ub = input_ub_;
    Q = q_weights_.asDiagonal();
    R = r_weights_.asDiagonal();
    R_inv = R.inverse();
    Qf = qf_weights_.asDiagonal();

    // LQR variables
    K_traj.resize(horizon + 1);
    S_xx.resize(horizon + 1);
    s_x.resize(horizon + 1);
}

void LQR::set_dynamics(StewartState &stewart_state) {
    set_dynamics(stewart_state.model.koopman_model.K_z, stewart_state.model.koopman_model.K_u);
}

void LQR::set_dynamics(const Eigen::MatrixXd &A_, const Eigen::MatrixXd &B_) {
    A = A_;
    B = B_;
}

void LQR::set_target(StewartState &stewart_state) {
    set_target(stewart_state.ctrl.ball_obs_d, stewart_state.ctrl.servo_cmd_d);
}

void LQR::set_target(Eigen::VectorXd &state_d_, Eigen::VectorXd &input_d_) {
    state_d = state_d_;
    input_d = input_d_;
}

// Continuous-time finite-horizon time-invariant LQR
// This function calculates all the gains based on {horizon, Q, R, Qf, state_d, input_d, A, B}
// https://underactuated.mit.edu/lqr.html#finite_horizon Chapter 8.2.4
void LQR::cal_gains() {
    // Reinitialize process variables and gains
    S_xx.clear();
    s_x.clear();
    K_traj.clear();
    S_xx.resize(horizon + 1);
    s_x.resize(horizon + 1);
    K_traj.resize(horizon + 1);

    // Terminal conditions
    S_xx[horizon] = Qf;
    s_x[horizon] = -Qf * state_d;
    K_traj[horizon] = R_inv * B.transpose() * S_xx[horizon];

    // Solve
    for (size_t i = horizon; i > 0; i--) {
        // Tmp variables
        const MatrixXd S_xx_t = S_xx[i];
        const VectorXd s_x_t = s_x[i];
        MatrixXd S_xx_dot;
        VectorXd s_x_dot;

        // HJB
        S_xx_dot = -(Q - S_xx_t * B * R_inv * B.transpose() * S_xx_t
                     + S_xx_t * A + A.transpose() * S_xx_t);
        s_x_dot = -(-Q * state_d + (A.transpose() - S_xx_t * B * R_inv * B.transpose()) * s_x_t
                    + S_xx_t * B * input_d);

        // Backward Euler
        S_xx[i - 1] = S_xx_t - S_xx_dot * dt;
        s_x[i - 1] = s_x_t - s_x_dot * dt;
        K_traj[i - 1] = R_inv * B.transpose() * S_xx[i - 1];
    }
}

void LQR::cal_action(StewartState &stewart_state) {
    update(stewart_state);
    cal_action(stewart_state.fbk.ball_obs);
}

void LQR::cal_action(const VectorXd &state) {
    // Calculate all control gains
    cal_gains();

    // Get the first input in the solution
    optimal_action = input_d - K_traj[0] * state - R_inv * B.transpose() * s_x[0];
}

void LQR::cal_action_for_sac(StewartState &stewart_state) {
    optimal_action =
            input_d - K_traj[0] * stewart_state.fbk.ball_obs - R_inv * B.transpose() * s_x[0];
}

//void LQR::simulate(StewartState &stewart_state, const size_t N, const Eigen::VectorXd &z0,
//                   vector<Eigen::VectorXd> &z_traj, vector<Eigen::VectorXd> &u_traj,
//                   vector<Eigen::MatrixXd> &df_dz_traj, vector<Eigen::MatrixXd> &df_du_traj,
//                   vector<Eigen::MatrixXd> &dmu_dz_traj) {
//    // Clear trajectory vectors
//    z_traj.clear();
//    u_traj.clear();
//    dmu_dz_traj.clear();
//    cout << "Traj clear!" << endl;
//
//    // Initialize
//    z_traj.push_back(z0);
//    VectorXd z = z0;
//    fill(df_dz_traj.begin(), df_dz_traj.end(), stewart_state.model.koopman_model.K_z);
//    cout << "df_dz_traj init!" << endl;
//    fill(df_du_traj.begin(), df_du_traj.end(), stewart_state.model.koopman_model.K_u);
//    cout << "df_du_traj init!" << endl;
//
//    auto ct_dyn = [&](const VectorXd &state, const VectorXd &input) {
//        return stewart_state.model.koopman_model.ct_dyn(state, input);
//    };
//    cout << "ct_dyn init!" << endl;
//
//    for (size_t i = 0; i < N; ++i) {
//        cal_action(z); // calculate optimal control
//        cout << "cal_action!" << endl;
//        u_traj.emplace_back(optimal_action);
//        dmu_dz_traj.emplace_back(K_traj[0]); // get control gain
//        cout << "emplace_back!" << endl;
//        Utils::euler_step(ct_dyn, dt, z, optimal_action); // update z
//        z_traj.emplace_back(z);
//        cout << "euler_step!" << endl;
//    }
//}

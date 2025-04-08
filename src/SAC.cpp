#include "rkl_cpp/SAC.hpp"

void SAC::init(StewartState &stewart_state) {
    // Initialize the nominal policy (we use LQR here)
    lqr_policy.init(stewart_state);

    // Initialize the adjoint variable
    adjoint_var.reset(CONTROL_PERIOD / 1000.0, stewart_state.params.ctrl_horizon);

    // Initialize the objective
    task.set_params(stewart_state.params.q_weights, stewart_state.params.r_weights,
                    stewart_state.params.qf_weights, stewart_state.params.learn_weight);
    task.set_state_ref(stewart_state.ctrl.ball_obs_d);

    sac_R = stewart_state.params.sac_r_weights.asDiagonal();
    df_dz_traj.resize(stewart_state.params.ctrl_horizon + 1);
    df_du_traj.resize(stewart_state.params.ctrl_horizon + 1);
}

void SAC::update(StewartState &stewart_state) {
    // Update the nominal policy (dynamics and target)
    lqr_policy.update(stewart_state);

    // Reset the adjoint variable
    adjoint_var.reset(CONTROL_PERIOD / 1000.0, stewart_state.params.ctrl_horizon);

    // Reset the weights and control target
    task.set_state_ref(stewart_state.ctrl.ball_obs_d);

    // Reset process variables
    z_traj.clear();
    u_traj.clear();
    df_dz_traj.clear();
    df_du_traj.clear();
    dmu_dz_traj.clear();
    df_dz_traj.resize(stewart_state.params.ctrl_horizon + 1);
    df_du_traj.resize(stewart_state.params.ctrl_horizon + 1);
}

void SAC::simulate(StewartState &stewart_state) {
    // Before running this function, make sure "update" has been performed
    // Fill df_dz_traj and df_du_traj
    fill(df_dz_traj.begin(), df_dz_traj.end(), stewart_state.model.koopman_model.K_z);
    fill(df_du_traj.begin(), df_du_traj.end(), stewart_state.model.koopman_model.K_u);

    // Initialize z_traj
    z_traj.emplace_back(stewart_state.fbk.ball_obs); // initial obs
    VectorXd z = z_traj[0];

    // Dynamics
    auto ct_dyn = [&](const VectorXd &z, const VectorXd &u) {
        return stewart_state.model.koopman_model.ct_dyn(z, u);
    };

    // Calculates all the gains based on {horizon, Q, R, Qf, state_d, input_d, A, B}
    lqr_policy.cal_gains();
    for (int i = 0; i < stewart_state.params.ctrl_horizon; ++i) {
        // Get the current gain and feedforward term
        MatrixXd K;
        VectorXd fw;
        K = lqr_policy.K_traj[i];
        fw = -lqr_policy.R_inv * lqr_policy.B.transpose() * lqr_policy.s_x[i];

        // Get the current optimal action
        VectorXd u_star;
        u_star = lqr_policy.input_d - K * z + fw;

        // Update z
        Utils::euler_step(ct_dyn, CONTROL_PERIOD / 1000.0, z, u_star);

        // Store variables
        z_traj.emplace_back(z);
        u_traj.emplace_back(u_star);
        dmu_dz_traj.emplace_back(K);
    }
}

void SAC::cal_action(StewartState &stewart_state) {
    // Update the nominal policy, reset adjoint variable, target, and process variables
    update(stewart_state);

    // Simulate forward, update z_traj, u_traj, df_dz_traj, df_du_traj, dmu_dz_traj
    simulate(stewart_state);

    // Update dl_dz_traj, dl_du_traj
    task.get_linearization(z_traj, u_traj);

    // Update terminal condition: rho_f = dm_dz
    adjoint_var.rho_f = task.cal_dm_dx(z_traj.back());

    // Update rho_traj
    adjoint_var.simulate_adjoint(task.dl_dx_traj, task.dl_du_traj, df_dz_traj, df_du_traj,
                                 dmu_dz_traj);

    // Compute the nominal input
    // lqr_policy.cal_action(stewart_state);
    lqr_policy.cal_action_for_sac(stewart_state);

    // SAC
    optimal_action = -sac_R.inverse() * df_du_traj[0].transpose() * adjoint_var.rho_traj[0]
                     + lqr_policy.optimal_action;
}

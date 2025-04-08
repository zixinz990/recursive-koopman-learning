#include "rkl_cpp/Optim.hpp"

void Adjoint::reset(const double dt_, const size_t horizon_) {
    dt = dt_;
    horizon = horizon_;
    rho_f.resize(NUM_STATE_OBS);
    rho_f.setZero();
    rho_traj.clear();
    rho_traj.resize(horizon + 1);
}

void Adjoint::simulate_adjoint(vector<VectorXd> &dl_dx_traj, vector<VectorXd> &dl_du_traj,
                               vector<MatrixXd> &df_dx_traj, vector<MatrixXd> &df_du_traj,
                               vector<MatrixXd> &dmu_dx_traj) {
    assert(rho_traj.size() == horizon + 1);
    rho_traj[horizon] = rho_f; // terminal condition
    for (size_t i = horizon; i > 0; i--) {
        VectorXd drho_dt;
        drho_dt = cal_drho_dt(rho_traj[i], dl_dx_traj[i - 1], dl_du_traj[i - 1],
                              df_dx_traj[i - 1], df_du_traj[i - 1], dmu_dx_traj[i - 1]);
        rho_traj[i - 1] = rho_traj[i] - drho_dt * dt;
    }
}

VectorXd Adjoint::cal_drho_dt(VectorXd &rho, VectorXd &dl_dx, VectorXd &dl_du,
                              MatrixXd &df_dx, MatrixXd &df_du,
                              MatrixXd &dmu_dx) {
    VectorXd drho_dt;
    drho_dt = -(dl_dx + dmu_dx.transpose() * dl_du) - (df_dx + df_du * dmu_dx).transpose() * rho;
    return drho_dt;
}

void Objective::set_params(VectorXd &Q_weights, VectorXd &R_weights, VectorXd &Qf_weights,
                           double &learn_weight_) {
    Q = Q_weights.asDiagonal();
    R = R_weights.asDiagonal();
    Qf = Qf_weights.asDiagonal();
    learn_weight = learn_weight_;
}

void Objective::set_state_ref(const VectorXd &state_d_) {
    state_d = state_d_;
}

VectorXd Objective::cal_dl_dx(const VectorXd &x) {
    VectorXd dl_dx = Q * (x - state_d) - learn_weight * 2.0 * x / ((x.transpose() * x + 1e-9) *
                                                                   (x.transpose() * x + 1e-9));
    return dl_dx;
}

VectorXd Objective::cal_dl_du(const VectorXd &u) {
    // VectorXd dl_du = R * u;
    VectorXd dl_du = R * u - learn_weight * 2.0 * u / ((u.transpose() * u + 1e-9) *
                                                       (u.transpose() * u + 1e-9));
    return dl_du;
}

VectorXd Objective::cal_dm_dx(const Eigen::VectorXd &x) {
    VectorXd dm_dx = Qf * (x - state_d);
    return dm_dx;
}

void Objective::get_linearization(vector<Eigen::VectorXd> &x_traj,
                                  vector<Eigen::VectorXd> &u_traj) {
    dl_dx_traj.clear();
    dl_du_traj.clear();
    for (size_t i = 0; i < u_traj.size(); ++i) {
        dl_dx_traj.emplace_back(cal_dl_dx(x_traj[i]));
        dl_du_traj.emplace_back(cal_dl_du(u_traj[i]));
    }
}

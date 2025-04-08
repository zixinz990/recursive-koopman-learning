#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "rkl_cpp/ConstParams.hpp"

using namespace std;
using namespace Eigen;

class Adjoint {
public:
    Adjoint() = default;

    // Reset
    void reset(double dt, size_t horizon);

    // Simulate the adjoint variable trajectory
    void simulate_adjoint(vector<VectorXd> &dl_dx_traj, vector<VectorXd> &dl_du_traj,
                          vector<MatrixXd> &df_dx_traj, vector<MatrixXd> &df_du_traj,
                          vector<MatrixXd> &dmu_dx_traj);

    // Time derivative of the adjoint variable
    static VectorXd cal_drho_dt(VectorXd &rho, VectorXd &dl_dx, VectorXd &dl_du,
                                MatrixXd &df_dx, MatrixXd &df_du, MatrixXd &dmu_dx);

    VectorXd rho_f;
    vector<VectorXd> rho_traj;

private:
    double dt;
    size_t horizon;
};

class Objective {
public:
    Objective() = default;

    void set_params(VectorXd &Q_weights, VectorXd &R_weights, VectorXd &Qf_weights,
                    double &learn_weight);

    void set_state_ref(const VectorXd &state_d);

    // Jacobian of the running cost w.r.t. the state
    VectorXd cal_dl_dx(const VectorXd &x);

    // Jacobian of the running cost w.r.t. the input
    VectorXd cal_dl_du(const VectorXd &u);

    // Jacobian of the terminal cost w.r.t. the state
    VectorXd cal_dm_dx(const VectorXd &x);

    // Get Jacobians along a trajectory
    void get_linearization(vector<VectorXd> &x_traj, vector<VectorXd> &u_traj);

    vector<VectorXd> dl_dx_traj;
    vector<VectorXd> dl_du_traj;
    double learn_weight = 100.0;
private:
    MatrixXd Q, R, Qf;
    VectorXd state_d; // desired state
};

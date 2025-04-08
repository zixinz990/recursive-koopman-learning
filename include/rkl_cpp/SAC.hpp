#pragma once

#include "rkl_cpp/LQR.hpp"
#include "rkl_cpp/Optim.hpp"
#include "rkl_cpp/StewartState.hpp"

using namespace std;
using namespace Eigen;

class SAC {
public:
    SAC() = default;

    // Initialize from StewartState
    void init(StewartState &stewart_state);

    // Update from StewartState
    void update(StewartState &stewart_state);

    // Simulate using the nominal policy
    void simulate(StewartState &stewart_state);

    // Calculate the optimal action
    void cal_action(StewartState &stewart_state);

    LQR lqr_policy;          // nominal policy
    Adjoint adjoint_var;     // adjoint variable
    Objective task;          // objective function
    VectorXd optimal_action; // solution

private:
    MatrixXd sac_R;
    vector<VectorXd> z_traj;      // trajectory of observables
    vector<VectorXd> u_traj;      // trajectory of inputs
    vector<MatrixXd> df_dz_traj;  // trajectory of K_z
    vector<MatrixXd> df_du_traj;  // trajectory of K_u
    vector<MatrixXd> dmu_dz_traj; // trajectory of policy gradient w.r.t. to observable
};

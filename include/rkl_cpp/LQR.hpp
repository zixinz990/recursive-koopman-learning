#pragma once

#include "rkl_cpp/ConstParams.hpp"
#include "rkl_cpp/StewartState.hpp"

using namespace std;
using namespace Eigen;

class LQR {
public:
    LQR() = default;

    ~LQR() = default;

    // Initialize from StewartState
    void init(StewartState &stewart_state);

    // Update from StewartState
    void update(StewartState &stewart_state);

    // Set parameters
    void set_params(StewartState &stewart_state);

    void set_params(double dt, size_t horizon,
                    const VectorXd &input_lb, const VectorXd &input_ub,
                    const VectorXd &q_weights, const VectorXd &r_weights,
                    const VectorXd &qf_weights);

    // Set time-invariant linear dynamics
    void set_dynamics(StewartState &stewart_state);

    void set_dynamics(const MatrixXd &A, const MatrixXd &B);

    // Set target
    void set_target(StewartState &stewart_state);

    void set_target(VectorXd &state_d, VectorXd &input_d);

    // Calculate the optimal action
    void cal_action(StewartState &stewart_state);

    void cal_action(const VectorXd &state);

    void cal_action_for_sac(StewartState &stewart_state);

    // Calculate all control gains
    void cal_gains();

//    // Simulate forward using the current Koopman model (for SAC)
//    void simulate(StewartState &stewart_state, size_t N, const VectorXd &z0,
//                  vector<Eigen::VectorXd> &z_traj, vector<Eigen::VectorXd> &u_traj,
//                  vector<Eigen::MatrixXd> &df_dz_traj, vector<Eigen::MatrixXd> &df_du_traj,
//                  vector<Eigen::MatrixXd> &dmu_dz_traj);

    vector<MatrixXd> dmu_dx_traj; // trajectory of policy gradient w.r.t. to state
    vector<MatrixXd> K_traj;      // all control gains
    VectorXd state_d;         // desired state
    VectorXd input_d;         // input reference
    MatrixXd Q, R, R_inv, Qf; // weight matrices
    MatrixXd A, B;            // continuous-time linear dynamics
    vector<VectorXd> s_x;     // process variable
    VectorXd optimal_action;      // solution

private:
    double dt;                // time step, sec
    size_t horizon;           // horizon length
    VectorXd input_lb;        // input lower bound
    VectorXd input_ub;        // input upper bound
    vector<MatrixXd> S_xx;    // process variable
};

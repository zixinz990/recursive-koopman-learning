#pragma once

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "rkl_cpp/ConstParams.hpp"
#include "rkl_cpp/Utils.hpp"

using namespace std;
using namespace Eigen;

class Koopman {
public:
    explicit Koopman(double dt) { reset(dt); };

    void reset(double dt);

    void edmd(const VectorXd &x0, const VectorXd &u0, const VectorXd &x1);

    void update_model(const VectorXd &x0, const VectorXd &u0, const VectorXd &x1);

    // Update the ct model without calculating matrix log
    void update_model_test(const VectorXd &x0, const VectorXd &u0, const VectorXd &x1);

    VectorXd ct_dyn(const VectorXd &z0, const VectorXd &u0);

    MatrixXd Q, P, K_dt;

    MatrixXd K_z, K_u;

    MatrixXd Y0, Y1;

private:
    double dt = CONTROL_PERIOD;    

    string path = "/home/stewart/Documents/Dev/koopman_ws/src/active-koopman-cpp/init_koopman/";
    string Y0_file_path = path + "Y0.csv";
    string Y1_file_path = path + "Y1.csv";
    string Q_file_path = path + "Q.csv"; // discrete-time
    string P_file_path = path + "P.csv"; // discrete-time
    string K_dt_file_path = path + "K_dt.csv"; // discrete-time
    string K_z_file_path = path + "K_z.csv"; // continuous-time
    string K_u_file_path = path + "K_u.csv"; // continuous-time
};

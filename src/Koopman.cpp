#include "rkl_cpp/Koopman.hpp"

void Koopman::reset(double dt_) {
    dt = dt_;
    Y0 = Utils::read_matrix_from_csv(Y0_file_path);
    Y1 = Utils::read_matrix_from_csv(Y1_file_path);
    Q = Utils::read_matrix_from_csv(Q_file_path); // discrete-time
    P = Utils::read_matrix_from_csv(P_file_path); // discrete-time
    K_dt = Utils::read_matrix_from_csv(K_dt_file_path); // discrete-time
    K_z = Utils::read_matrix_from_csv(K_z_file_path); // continuous-time
    K_u = Utils::read_matrix_from_csv(K_u_file_path); // continuous-time
}

void Koopman::edmd(const VectorXd &x0, const VectorXd &u0, const VectorXd &x1) {
    double start = chrono::duration<double, milli>(
        chrono::system_clock::now().time_since_epoch()).count();
    VectorXd z0 = Utils::poly_obs(x0);
    VectorXd z1 = Utils::poly_obs(x1);

    // Expand z
    z0.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS);
    z1.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS);
    z0.tail(NUM_INPUT_OBS) = u0;
    z1.tail(NUM_INPUT_OBS) = u0;
    assert(z0.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
    assert(z1.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);

    // Update Y0 and Y1 using the new data
    MatrixXd Y0_old = Y0;
    MatrixXd Y1_old = Y1;
    int num_data = Y0.cols();
    assert(Y0.rows() == NUM_STATE_OBS + NUM_INPUT_OBS);
    assert(Y1.rows() == NUM_STATE_OBS + NUM_INPUT_OBS);
    Y0.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS, Y0.cols() + 1);
    Y1.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS, Y1.cols() + 1);
    Y0.col(Y0.cols() - 1) = z0;
    Y1.col(Y1.cols() - 1) = z1;
    assert(Y0.rows() == NUM_STATE_OBS + NUM_INPUT_OBS);
    assert(Y1.rows() == NUM_STATE_OBS + NUM_INPUT_OBS);
    assert(Y0.cols() == num_data + 1);
    assert(Y1.cols() == num_data + 1);

    // Check Y0 and Y1
    // num_data = Y0.cols();
    // MatrixXd diff_1 = Y0_old - Y0.leftCols(num_data - 1);
    // MatrixXd diff_2 = Y1_old - Y1.leftCols(num_data - 1);
    // MatrixXd diff_3 = Y0.col(num_data - 1) - z0;
    // MatrixXd diff_4 = Y1.col(num_data - 1) - z1;
    // cout << "diff_1.norm(): " << diff_1.norm() << endl;
    // cout << "diff_2.norm(): " << diff_2.norm() << endl;
    // cout << "diff_3.norm(): " << diff_3.norm() << endl;
    // cout << "diff_4.norm(): " << diff_4.norm() << endl;

    // EDMD
    MatrixXd tmp = Y0 * Y0.transpose();
    // K_dt = Y1 * Y0.transpose() * Utils::pseudo_inverse(tmp);
    K_dt = Y1 * Y0.transpose() * tmp.inverse();
    // MatrixXd K_ct = (K_dt.log()) / dt;
    MatrixXd K_ct = Utils::matrix_log(K_dt, 30);
    K_z = K_ct.topLeftCorner(NUM_STATE_OBS, NUM_STATE_OBS);
    K_u = K_ct.topRightCorner(NUM_STATE_OBS, NUM_INPUT_OBS);

    double end = chrono::duration<double, milli>(
        chrono::system_clock::now().time_since_epoch()).count();
    cout << "(Retrain) The Koopman model takes " << end - start << " ms to update" << endl;
}

void Koopman::update_model(const VectorXd &x0, const VectorXd &u0, const VectorXd &x1) {
//    cout << "Updating the Koopman model..." << endl;
    double start = chrono::duration<double, milli>(
        chrono::system_clock::now().time_since_epoch()).count();
    VectorXd z0 = Utils::poly_obs(x0);
    VectorXd z1 = Utils::poly_obs(x1);

    // Expand z
    z0.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS);
    z1.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS);
    z0.tail(NUM_INPUT_OBS) = u0;
    z1.tail(NUM_INPUT_OBS) = u0;
    assert(z0.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
    assert(z1.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);

    // Update the discrete-time Koopman operator
    double gamma = 1.0 / (1.0 + z0.transpose() * P * z0);
    K_dt += gamma * (z1 - K_dt * z0) * z0.transpose() * P; // Update K_dt
    assert(K_dt.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
    assert(K_dt.cols() == NUM_STATE_OPS + NUM_INPUT_OBS);
    P -= gamma * P * z0 * z0.transpose() * P; // Update P

    // Calculate the continuous-time Koopman operator
    MatrixXd K_ct = (K_dt.log()) / dt;
    // MatrixXd K_ct = Utils::matrix_log(K_dt, 30);
    assert(K_ct.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
    assert(K_ct.cols() == NUM_STATE_OPS + NUM_INPUT_OBS);
    K_z = K_ct.topLeftCorner(NUM_STATE_OBS, NUM_STATE_OBS);
    K_u = K_ct.topRightCorner(NUM_STATE_OBS, NUM_INPUT_OBS);

    double end = chrono::duration<double, milli>(
        chrono::system_clock::now().time_since_epoch()).count();
    cout << "(RLS) The Koopman model takes " << end - start << " ms to update" << endl;
}

// void Koopman::update_model_test(const VectorXd &x0, const VectorXd &u0, const VectorXd &x1) {
//     // Initial CT Koopman matrix
//     MatrixXd K_ct = (K_dt.log()) / dt;

//     double start = chrono::duration<double, milli>(
//         chrono::system_clock::now().time_since_epoch()).count();
//     VectorXd z0 = Utils::poly_obs(x0);
//     VectorXd z1 = Utils::poly_obs(x1);

//     // Expand z
//     z0.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS);
//     z1.conservativeResize(NUM_STATE_OBS + NUM_INPUT_OBS);
//     z0.tail(NUM_INPUT_OBS) = u0;
//     z1.tail(NUM_INPUT_OBS) = u0;
//     assert(z0.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
//     assert(z1.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);

//     // Estimate dz/dt
//     VectorXd zdot = (z1 - z0) / dt;

//     // Update CT Koopman
//     double gamma = 1.0 / (1.0 + z0.transpose() * P * z0);
//     K_ct += gamma * (zdot - K_ct * z0) * z0.transpose() * P; // Update K_dt
//     assert(K_ct.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
//     assert(K_ct.cols() == NUM_STATE_OPS + NUM_INPUT_OBS);
//     P -= gamma * P * z0 * z0.transpose() * P; // Update P

//     assert(K_ct.rows() == NUM_STATE_OPS + NUM_INPUT_OBS);
//     assert(K_ct.cols() == NUM_STATE_OPS + NUM_INPUT_OBS);
//     K_z = K_ct.topLeftCorner(NUM_STATE_OBS, NUM_STATE_OBS);
//     K_u = K_ct.topRightCorner(NUM_STATE_OBS, NUM_INPUT_OBS);

//     double end = chrono::duration<double, milli>(
//         chrono::system_clock::now().time_since_epoch()).count();
//     cout << "(RLS test) The Koopman model takes " << end - start << " ms to update" << endl;
// }

VectorXd Koopman::ct_dyn(const VectorXd &z0, const VectorXd &u0) {
    return K_z * z0 + K_u * u0;
}

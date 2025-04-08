#include "rkl_cpp/Utils.hpp"

// Each column is a center, in cm
MatrixXd Utils::rbf_centers = 100.0 * Utils::read_matrix_from_csv(
        "/home/stewart/Documents/Dev/koopman_ws/src/active-koopman-cpp/init_koopman/rbf_centers.csv").transpose();

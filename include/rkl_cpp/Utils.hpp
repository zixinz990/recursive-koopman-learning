#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include "rkl_cpp/ConstParams.hpp"
#include "rkl_cpp/Koopman.hpp"

using namespace std;
using namespace Eigen;

using ct_dyn_function = function<VectorXd(VectorXd, VectorXd)>;


class Utils {
public:
    static VectorXd poly_obs(const VectorXd &x) {
        VectorXd z(NUM_STATE_OBS);
        z << x(0), x(1), x(2), x(3),
                x(0) * x(0), x(1) * x(1), x(2) * x(2), x(3) * x(3),
                x(0) * x(1), x(0) * x(2), x(0) * x(3),
                x(1) * x(2), x(1) * x(3), x(2) * x(3),
                x(0) * x(0) * x(0),
                x(1) * x(1) * x(1),
                x(2) * x(2) * x(2),
                x(3) * x(3) * x(3),
                x(0) * x(0) * x(1), x(0) * x(0) * x(2), x(0) * x(0) * x(3),
                x(1) * x(1) * x(2), x(1) * x(1) * x(3),
                x(2) * x(2) * x(3),
                x(0) * x(1) * x(2), x(0) * x(1) * x(3),
                x(0) * x(2) * x(3), x(1) * x(2) * x(3);

        assert(z.rows() == NUM_STATE_OBS && "Error: Dimension of z does not match NUM_STATE_OBS");
        return z;
    }

    static VectorXd fourier_obs(const VectorXd &x) {
        VectorXd z(NUM_STATE_OBS);
        // z << x(0), x(1), x(2), x(3),
        //      1, 1, 1, 1,
        //      sin(x(0)), sin(x(1)), sin(x(2)), sin(x(3)),
        //      cos(x(0)), cos(x(1)), cos(x(2)), cos(x(3)),
        //      sin(2*x(0)), sin(2*x(1)), sin(2*x(2)), sin(2*x(3)),
        //      cos(2*x(0)), cos(2*x(1)), cos(2*x(2)), cos(2*x(3)),
        //      sin(3*x(0)), sin(3*x(1)), sin(3*x(2)), sin(3*x(3)),
        //      cos(3*x(0)), cos(3*x(1)), cos(3*x(2)), cos(3*x(3));
        z << x(0), x(1), x(2), x(3),
             1, 1, 1, 1,
             sin(x(0)), sin(x(1)), sin(x(2)), sin(x(3)),
             sin(2*x(0)), sin(2*x(1)), sin(2*x(2)), sin(2*x(3)),
             sin(3*x(0)), sin(3*x(1)), sin(3*x(2)), sin(3*x(3)),
             sin(4*x(0)), sin(4*x(1)), sin(4*x(2)), sin(4*x(3)),
             sin(5*x(0)), sin(5*x(1)), sin(5*x(2)), sin(5*x(3)),
             sin(6*x(0)), sin(6*x(1)), sin(6*x(2)), sin(6*x(3));

        assert(z.rows() == NUM_STATE_OBS && "Error: Dimension of z does not match NUM_STATE_OBS");
        return z;
    }

    static VectorXd gaussian_rbf(const VectorXd &x) {
        double eps = 0.001;
        assert(rbf_centers.rows() == 4);
        assert(rbf_centers.cols() == NUM_STATE_OBS - 4);

        VectorXd z(NUM_STATE_OBS);
        z.head(4) = x;

        for (int i = 0; i < rbf_centers.cols(); ++i) {
            VectorXd d = x - rbf_centers.col(i);
            z(i + 4) = exp(-eps * d.squaredNorm());
        }

        return z;
    }

    static MatrixXd read_matrix_from_csv(const string &mat_file_path) {
        ifstream file(mat_file_path);
        vector<vector<double>> data;
        string line, cell;

        // Read the file line by line
        if (file.is_open()) {
            while (getline(file, line)) {
                stringstream line_stream(line);
                vector<double> row;
                while (getline(line_stream, cell, ',')) {
                    row.push_back(stod(cell));
                }
                data.push_back(row);
            }
            file.close();
        } else {
            // Return an empty matrix if file reading fails
            cerr << "Error: Could not open the file for reading: " << mat_file_path << endl;
            return MatrixXd();
        }

        // Return an empty matrix if there's no data
        if (data.size() == 0 || data[0].size() == 0) return MatrixXd();

        // Convert the vector to an Eigen matrix
        MatrixXd mat(data.size(), data[0].size());
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                mat(i, j) = data[i][j];
            }
        }
        return mat;
    }

    static void euler_step(const ct_dyn_function &ct_dyn, const double dt,
                           VectorXd &state, const VectorXd &input) {
        state += ct_dyn(state, input) * dt;
    }

    static void rk4_step(ct_dyn_function &ct_dyn, const double dt,
                         VectorXd &state, const VectorXd &input) {
        VectorXd k1 = ct_dyn(state, input);
        VectorXd k2 = ct_dyn(state + dt * k1 / 2.0, input);
        VectorXd k3 = ct_dyn(state + dt * k2 / 2.0, input);
        VectorXd k4 = ct_dyn(state + dt * k3, input);
        state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    }

    static vector<VectorXd> cal_slopes(const vector<double> &x, const vector<VectorXd> &y) {
        vector<VectorXd> slopes;
        slopes.reserve(x.size() - 1);
        for (size_t i = 0; i < x.size() - 1; ++i) {
            slopes.emplace_back((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
        }
        return slopes;
    }

    static vector<VectorXd> interp1d(const vector<double> &x, const vector<VectorXd> &y,
                                     const vector<double> &x_val) {
        if (x.size() != y.size()) {
            throw invalid_argument("x and y vectors must be of the same size.");
        }
        if (x.size() < 2)
            throw invalid_argument("Need at least two data points for interpolation.");

        // Precompute slopes for each interval in x
        vector<VectorXd> slopes = cal_slopes(x, y);

        // Prepare the output vector for interpolated values
        vector<VectorXd> y_interp;
        y_interp.reserve(x_val.size());

        size_t i = 0; // Index for x intervals

        // Iterate over each value in x_val to interpolate
        for (const auto &x_v: x_val) {
            // Check if x_v is within the interpolation range
            if (x_v < x.front() || x_v > x.back())
                throw out_of_range("x_val element is out of the interpolation range.");

            // Move to the correct interval
            while (i < x.size() - 2 && x_v > x[i + 1]) {
                ++i;
            }

            // Use precomputed slope for the interpolation
            VectorXd interpolated_value = y[i] + slopes[i] * (x_v - x[i]);
            y_interp.push_back(interpolated_value);
        }

        return y_interp;
    }

    // Function to compute Moore-Penrose pseudoinverse using SVD
    static MatrixXd pseudo_inverse(const MatrixXd& A, double tolerance = 1e-6) {
        // Perform SVD
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

        // Extract the singular values (as a vector)
        const VectorXd& singularValues = svd.singularValues();

        // Prepare an appropriately sized diagonal matrix for the inverted singular values
        VectorXd singularValuesInv(singularValues.size());
        singularValuesInv.setZero();

        // Invert the singular values, considering the tolerance
        for (int i = 0; i < singularValues.size(); ++i) {
            if (singularValues(i) > tolerance) {
                singularValuesInv(i) = 1.0 / singularValues(i);
            } else {
                singularValuesInv(i) = 0.0;  // Value below tolerance, treat as 0
            }
        }

        // Construct the pseudo-inverse of the diagonal matrix S
        MatrixXd S_plus = MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().cols());
        for (int i = 0; i < singularValuesInv.size(); ++i) {
            S_plus(i, i) = singularValuesInv(i);
        }

        // A^+ = V * S_plus * U^T        
        MatrixXd A_pinv = svd.matrixV() * S_plus * svd.matrixU().transpose();

        return A_pinv;
    }

    static MatrixXd matrix_log(const MatrixXd& A, const int degree) {
        MatrixXd I = MatrixXd::Identity(A.rows(), A.cols());
        MatrixXd X = I - A;
        MatrixXd result = -X;
        MatrixXd X_power = X;
        double factorial = 1.0;
        for (size_t i = 2; i <= degree; ++i) {
            factorial += 1.0;
            X_power *= X;
            result -= X_power / factorial;
        }
        return result;
    }

    static MatrixXd rbf_centers;
};

template<typename T>
class FixedSizeQueue {
public:
    explicit FixedSizeQueue(size_t max_size_) { max_size = max_size_; }

    void push(const T &value) {
        // Remove the oldest element (the first one)
        if (vec.size() == max_size) vec.erase(vec.begin());
        vec.emplace_back(value);
    }

    void empty() {
        vec.clear();
    }

    bool is_full() {
        return vec.size() == max_size;
    }

    T cal_avg(size_t window_size) {
        vector<T> vec_tmp(vec.end() - window_size, vec.end());
        T sum;
        for (size_t i = 0; i < vec_tmp.size(); ++i) {
            sum += vec[i];
        }
        return sum / vec_tmp.size();
    }

    vector<T> vec;

private:
    size_t max_size;
};

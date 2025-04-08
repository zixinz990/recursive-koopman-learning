#include <chrono>
#include <memory>
#include <iostream>
#include <mutex>
#include <thread>

#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"

#include "rkl_cpp/ConstParams.hpp"
#include "rkl_cpp/Interface.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) {
    // Initialize ROS
    rclcpp::init(argc, argv);
    auto node = make_shared<rclcpp::Node>("rkl_node");

    // Initialize interface
    cout << "Initializing the interface..." << endl;
    auto mtx_ptr = make_shared<mutex>();
    unique_ptr<Interface> interface = std::make_unique<Interface>(node, mtx_ptr);
    rclcpp::sleep_for(chrono::nanoseconds(5000000000));
//    Interface interface = Interface(node);

    //////////////////////////
    /// THREAD 1: FEEDBACK ///
    //////////////////////////
    thread feedback_thread([&]() {
        cout << "Entering the feedback thread..." << endl;
        while (rclcpp::ok()) {
            auto start = chrono::high_resolution_clock::now();
            bool fbk_running = interface->fbk_update();

            if (!fbk_running) {
                cerr << "Feedback thread is terminated because of errors." << endl;
                rclcpp::shutdown();
                break;
            }
            this_thread::sleep_until(
                    start + chrono::microseconds(static_cast<int>(FEEDBACK_PERIOD * 1000.0)));
        }
    });

    ///////////////////////////
    /// THREAD 2: SERVO CMD ///
    ///////////////////////////
    thread servo_cmd_thread([&]() {
        cout << "Entering the servo commands thread..." << endl;
        while (rclcpp::ok()) {
            auto start = chrono::high_resolution_clock::now();
            bool servo_cmd_sending = interface->send_cmd();

            if (!servo_cmd_sending) {
                cerr << "The servo commands were terminated because of errors." << endl;
                rclcpp::shutdown();
                break;
            }
            this_thread::sleep_until(
                    start + chrono::microseconds(static_cast<int>(SERVO_CMD_PERIOD * 1000.0)));
        }
    });

    //////////////////////////////
    /// THREAD 3: MODEL UPDATE ///
    //////////////////////////////
    thread model_update_thread([&]() {
        cout << "Entering the model update thread..." << endl;
        while (rclcpp::ok()) {
            auto start = chrono::high_resolution_clock::now();
            bool model_updating = interface->model_update();

            if (!model_updating) {
                cerr << "The model is not updating because of errors." << endl;
                rclcpp::shutdown();
                break;
            }
            this_thread::sleep_until(
                    start + chrono::microseconds(static_cast<int>(MODEL_UPDATE_PERIOD * 1000.0)));
        }
    });

    /////////////////////////
    /// THREAD 4: CONTROL ///
    /////////////////////////
    thread control_thread([&]() {
        cout << "Entering the controller thread..." << endl;
        while (rclcpp::ok()) {
            auto start = chrono::high_resolution_clock::now();
            bool control_looping = interface->ctrl_update();

            if (!control_looping) {
                cerr << "The controller is terminated because of errors." << endl;
                rclcpp::shutdown();
                break;
            }
            this_thread::sleep_until(
                    start + chrono::microseconds(static_cast<int>(CONTROL_PERIOD * 1000.0)));
        }
    });

    rclcpp::spin(node);
    feedback_thread.join();
    servo_cmd_thread.join();
    model_update_thread.join();
    control_thread.join();

    return 0;
}

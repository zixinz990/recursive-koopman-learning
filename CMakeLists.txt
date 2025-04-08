cmake_minimum_required(VERSION 3.23)
project(rkl_cpp)
set(CMAKE_BUILD_TYPE Release)

# Default to C99
if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

############################
####### DEPENDENCIES #######
############################

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(platform_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ifopt REQUIRED)
find_package(OsqpEigen REQUIRED)

include_directories(include)

#########################
####### LIBRARIES #######
#########################

add_library(rkl_lib
        src/Interface.cpp
        src/Koopman.cpp
        src/Optim.cpp
        src/StewartState.cpp
        src/LQR.cpp
        src/SAC.cpp
        src/Utils.cpp
#        tests/test_ifopt.cpp
)
target_link_libraries(rkl_lib
        Eigen3::Eigen
)
ament_target_dependencies(rkl_lib rclcpp platform_interfaces)

#########################
###### EXECUTABLES ######
#########################

add_executable(rkl_node src/Main.cpp)
target_link_libraries(rkl_node
        rkl_lib
        Eigen3::Eigen
)
ament_target_dependencies(rkl_node rclcpp)
install(TARGETS
        rkl_node
        DESTINATION lib/${PROJECT_NAME}
)

###################
###### TESTS ######
###################

#add_executable(test_lqr_double_integrator tests/test_lqr_double_integrator.cpp)
#target_link_libraries(test_lqr_double_integrator
#        rkl_lib
#        Eigen3::Eigen
#)
#ament_target_dependencies(test_lqr_double_integrator rclcpp)
#install(TARGETS
#        test_lqr_double_integrator
#        DESTINATION lib/${PROJECT_NAME}
#)
#
#add_executable(test_interp tests/test_interp.cpp)
#target_link_libraries(test_interp
#        rkl_lib
#        Eigen3::Eigen
#)
#ament_target_dependencies(test_interp rclcpp)
#install(TARGETS
#        test_interp
#        DESTINATION lib/${PROJECT_NAME}
#)
#
#add_executable(test_ros tests/test_ros.cpp)
#target_link_libraries(test_ros
#        rkl_lib
#        Eigen3::Eigen
#)
#ament_target_dependencies(test_ros rclcpp)
#install(TARGETS
#        test_ros
#        DESTINATION lib/${PROJECT_NAME}
#)
#
#add_executable(test_ifopt tests/test_ifopt.cpp)
#target_link_libraries(test_ifopt
#        rkl_lib
#        ifopt::ifopt_ipopt
#        Eigen3::Eigen
#)
#ament_target_dependencies(test_ifopt rclcpp)
#install(TARGETS
#        test_ifopt
#        DESTINATION lib/${PROJECT_NAME}
#)
#
#add_executable(test_osqp_eigen tests/test_osqp_eigen.cpp)
#target_link_libraries(test_osqp_eigen
#        rkl_lib
#        ifopt::ifopt_ipopt
#        Eigen3::Eigen
#        OsqpEigen::OsqpEigen
#)
#ament_target_dependencies(test_osqp_eigen rclcpp)
#install(TARGETS
#        test_osqp_eigen
#        DESTINATION lib/${PROJECT_NAME}
#)

# add_executable(test_gaussian_rbf tests/test_gaussian_rbf.cpp)
# target_link_libraries(test_gaussian_rbf
#         rkl_lib
#         Eigen3::Eigen
#         OsqpEigen::OsqpEigen
# )
# ament_target_dependencies(test_gaussian_rbf rclcpp)
# install(TARGETS
#         test_gaussian_rbf
#         DESTINATION lib/${PROJECT_NAME}
# )

# add_executable(test_matrix_log tests/test_matrix_log.cpp)
# target_link_libraries(test_matrix_log
#         rkl_lib
#         Eigen3::Eigen
#         OsqpEigen::OsqpEigen
# )
# ament_target_dependencies(test_matrix_log rclcpp)
# install(TARGETS
#         test_matrix_log
#         DESTINATION lib/${PROJECT_NAME}
# )

# add_executable(test_pseudo_inverse tests/test_pseudo_inverse.cpp)
# target_link_libraries(test_pseudo_inverse
#         rkl_lib
#         Eigen3::Eigen
#         OsqpEigen::OsqpEigen
# )
# ament_target_dependencies(test_pseudo_inverse rclcpp)
# install(TARGETS
#         test_pseudo_inverse
#         DESTINATION lib/${PROJECT_NAME}
# )

ament_package()

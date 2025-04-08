import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare arguments
    task_arg = DeclareLaunchArgument(
        'control_task',
        default_value='0',
        description='Control task: puck balancing (0) or trajectory tracking (1)'
    )
    obs_arg = DeclareLaunchArgument(
        'observation_function',
        default_value='0',
        description='Observation function: polynomial observables (0) or Gaussian RBFs (1)'
    )
    mpc_type_arg = DeclareLaunchArgument(
        'mpc_type',
        default_value='0',
        description='MPC type: SAC (0) or LQR (1)'
    )
    model_update_arg = DeclareLaunchArgument(
        'model_update',
        default_value='0',
        description='Enable online model updating: no (0) or yes (1)'
    )    

    # Get configurations
    control_task = LaunchConfiguration('control_task')
    observation_function = LaunchConfiguration('observation_function')
    model_update = LaunchConfiguration('model_update')
    mpc_type = LaunchConfiguration('mpc_type')

    # Define mappings
    task_map = {'0': 'balance', '1': 'track_traj'}
    obs_map = {'0': 'poly', '1': 'rbf'}
    mpc_map = {'0': 'sac', '1': 'lqr'}
    update_map = {'0': 'kl', '1': 'rkl'}
    
    # Find the configuration file
    package_dir = get_package_share_directory('rkl_cpp')
    param_file_path = os.path.join(
        package_dir,
        'config',
        task_map[control_task.perform(None)] + '_' +
        obs_map[observation_function.perform(None)] + '_' +
        mpc_map[mpc_type.perform(None)] + '_' +
        update_map[online_update.perform(None)] + '_params.yaml'
    )

    # Launch nodes
    rkl_node = Node(
        package='rkl_cpp',
        executable='rkl_node',
        name='rkl_node',
        output='screen',
        parameters=[param_file_path]
    )

    motor_management_node = Node(
        package='motor_management',
        executable='motor_manage'
    )

    return LaunchDescription([rkl_node, motor_management_node])

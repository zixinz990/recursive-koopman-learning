from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

REPO_PATH = '/home/stewart/Downloads/soft_stewart'
PKG_PATH = REPO_PATH + '/stewart_ros/src/koopman_mpc'
SERVO_ANGLES_MSG_PATH = REPO_PATH + '/stewart_ros/src/platform_interfaces/msg/ServoAngles.msg'
BALL_ODOMETRY_MSG_PATH = REPO_PATH + '/stewart_ros/src/platform_interfaces/msg/BallOdometry.msg'
MOTOR_ODOM_MSG_PATH = REPO_PATH + '/stewart_ros/src/platform_interfaces/msg/Odom.msg'

class BallPosDataProcessor:
    def __init__(self, rosbag_name_list, dt, x_delay, u_delay, p_maf_size, v_maf_size, u_maf_size,
                 use_vel_est=False, use_u_traj=True, do_plot=False):
        """

        This class is used to generate the dataset used to train the Stewart platform control model

        Args:
            rosbag_name_list:   a list containing the names of ROS bags
            dt: time step       in second
            x_delay:            state delay
            u_delay:            input delay
            p_maf_size:         the window size of the moving average filter for processing the interpolated position data
            v_maf_size:         the window size of the moving average filter for processing the interpolated velocity data
            u_maf_size:         the window size of the moving average filter for processing the raw control data
            use_vel_est:        whether to use velocity estimation data
            use_u_traj:         whether to use a trajectory of control input
            do_plot:            whether to plot

        """
        # Load arguments
        self.rosbag_name_list = rosbag_name_list
        self.dt = dt
        self.x_delay = x_delay
        self.u_delay = u_delay
        self.p_maf_size = p_maf_size
        self.v_maf_size = v_maf_size
        self.u_maf_size = u_maf_size
        self.use_vel_est = use_vel_est
        self.use_u_traj = use_u_traj
        self.do_plot = do_plot

        self.p_dim = 2  # position vector dimension
        self.v_dim = 2  # velocity vector dimension
        self.x_dim = self.p_dim + self.v_dim if self.use_vel_est else self.p_dim  # state vector dimension
        self.u_dim = 6  # control vector dimension
        self.r_dim = self.u_dim  # servos positions vector dimension

        self.q_dim = self.x_dim * (self.x_delay + 1)  # state history vector dimension
        self.w_dim = self.u_dim * (self.u_delay + 1) if self.use_u_traj else self.u_dim  # control history vector dimension
        self.s_dim = self.w_dim  # servos positions history vector dimension

        # Initialize dataset
        self.dataset = {'q': [], 'q_next': [], 'w': [], 's': [], 'x': [], 'x_next': [], 'u': [], 'bag_start_idx': []}

    def get_traj_example(self, do_plot, start_idx, traj_len):
        q_traj = np.array(self.dataset['q'][start_idx:start_idx + traj_len])
        w_traj = np.array(self.dataset['w'][start_idx:start_idx + traj_len])
        s_traj = np.array(self.dataset['s'][start_idx:start_idx + traj_len])
        x_traj = np.array(self.dataset['x'][start_idx:start_idx + traj_len])
        u_traj = np.array(self.dataset['u'][start_idx:start_idx + traj_len])

        if do_plot:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            axs.plot(x_traj[:, 0], x_traj[:, 1], '.-')
            axs.plot(x_traj[0, 0], x_traj[0, 1], 'r*', markersize=12, label='First Point')
            axs.set_aspect('equal')
            axs.legend()
            fig.suptitle('Position Trajectory Example')

            fig, axs = plt.subplots(2, 1, figsize=(20, 10))
            axs[0].plot(x_traj[:, 2])
            axs[0].set_ylabel('X Velocity')
            axs[1].plot(x_traj[:, 3])
            axs[1].set_ylabel('Y Velocity')
            fig.suptitle('Velocity Trajectory Example')

            fig, axs = plt.subplots(6, 1, figsize=(20, 15))
            for i in range(self.u_dim):
                axs[i].plot(u_traj[:, i])
            fig.suptitle('Servo Command Trajectory Example')

        return q_traj, w_traj, s_traj, x_traj, u_traj

    def generate_dataset(self, desired_num_data):
        # Get message format
        servo_angles_msg_text = Path(SERVO_ANGLES_MSG_PATH).read_text()
        ball_odometry_msg_text = Path(BALL_ODOMETRY_MSG_PATH).read_text()
        motor_odom_msg_text = Path(MOTOR_ODOM_MSG_PATH).read_text()

        # Add the ServoAngle message format to the type store
        type_store = get_typestore(Stores.LATEST)
        add_types = {}
        add_types.update(get_types_from_msg(servo_angles_msg_text, 'platform_interfaces/msg/ServoAngles'))
        add_types.update(get_types_from_msg(ball_odometry_msg_text, 'platform_interfaces/msg/BallOdometry'))
        add_types.update(get_types_from_msg(motor_odom_msg_text, 'platform_interfaces/msg/Odom'))
        type_store.register(add_types)

        # Iterate over ROS bags
        for rosbag in self.rosbag_name_list:
            with Reader(rosbag) as reader:
                self.dataset['bag_start_idx'].append(len(self.dataset['q']))
                self.process_one_rosbag(type_store, reader)

        # Choose some data
        num_data = len(self.dataset['x'])
        if desired_num_data < num_data:
            self.dataset['q'] = np.array(self.dataset['q'])[:desired_num_data]
            self.dataset['q_next'] = np.array(self.dataset['q_next'])[:desired_num_data]
            self.dataset['x'] = np.array(self.dataset['x'])[:desired_num_data]
            self.dataset['x_next'] = np.array(self.dataset['x_next'])[:desired_num_data]
            self.dataset['u'] = np.array(self.dataset['u'])[:desired_num_data]
            self.dataset['w'] = np.array(self.dataset['w'])[:desired_num_data]
            self.dataset['s'] = np.array(self.dataset['s'])[:desired_num_data]
        else:
            print('Desired number of data is larger than the total number of data')

        # Plot all positions
        # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        # dataset_x = np.array(self.dataset['x'])
        # axs.plot(dataset_x[:, 0], dataset_x[:, 1], '.', alpha=0.15)
        # axs.set_aspect('equal')

    def process_one_rosbag(self, type_store, reader):
        """
        Read one ROS bag, get raw data, interpolate, and generate dataset.
        """
        p_data_raw = []
        p_filtered_data_raw = []
        v_data_raw = []
        v_filtered_data_raw = []
        u_data_raw = []
        motor_pos_data_raw = []

        p_time_raw = []
        p_filtered_time_raw = []
        v_time_raw = []
        v_filtered_time_raw = []
        u_time_raw = []
        motor_pos_time_raw = []

        # Iterate over messages
        for connection, timestamp, raw_data in reader.messages():
            # Get timestamp (second)
            time_curr = timestamp / 1e9

            # Get raw state data and timestamps
            if connection.topic == '/ball_pose':
                msg = type_store.deserialize_cdr(raw_data, connection.msgtype)
                p_time_raw.append(time_curr)
                p_data_raw.append(np.array([msg.position.x, msg.position.y]))

            # Get filtered ball pose data
            if connection.topic == '/filtered_ball_pose':
                msg = type_store.deserialize_cdr(raw_data, connection.msgtype)
                p_filtered_time_raw.append(time_curr)
                p_filtered_data_raw.append(np.array([msg.x, msg.y]))

            # Get filtered ball velocity data
            if connection.topic == '/filtered_ball_pose':
                msg = type_store.deserialize_cdr(raw_data, connection.msgtype)
                v_filtered_time_raw.append(time_curr)
                v_filtered_data_raw.append(np.array([msg.xdot, msg.ydot]))

            # Get raw control data and timestamps
            if connection.topic == '/servo_angles':
                msg = type_store.deserialize_cdr(raw_data, connection.msgtype)
                u_time_raw.append(time_curr)
                u_data_raw.append(np.array([msg.angle_1, msg.angle_2, msg.angle_3, msg.angle_4, msg.angle_5, msg.angle_6]))

            # Get raw motor position data and timestamps
            if connection.topic == '/motor_positions':
                msg = type_store.deserialize_cdr(raw_data, connection.msgtype)
                motor_pos_time_raw.append(time_curr)
                motor_pos_data_raw.append(msg.data * 2.44e-4 * 360.0)

        # Estimate velocity (using raw position data)
        for i in range(1, len(p_data_raw)):
            p_curr = p_data_raw[i]
            p_prev = p_data_raw[i - 1]
            t_curr = p_time_raw[i]
            t_prev = p_time_raw[i - 1]
            v_est = (p_curr - p_prev) / (t_curr - t_prev)
            v_time_raw.append(t_curr)
            v_data_raw.append(v_est)

        # List to array
        p_data_raw = np.array(p_data_raw)
        p_filtered_data_raw = np.array(p_filtered_data_raw)
        v_data_raw = np.array(v_data_raw)
        v_filtered_data_raw = np.array(v_filtered_data_raw)
        u_data_raw = np.array(u_data_raw)
        motor_pos_data_raw = np.array(motor_pos_data_raw)
        p_time_raw = np.array(p_time_raw)
        p_filtered_time_raw = np.array(p_filtered_time_raw)
        v_time_raw = np.array(v_time_raw)
        v_filtered_time_raw = np.array(v_filtered_time_raw)
        u_time_raw = np.array(u_time_raw)
        motor_pos_time_raw = np.array(motor_pos_time_raw)

        # Motor pos raw data process
        motor_pos_data_raw[:, 0] = -motor_pos_data_raw[:, 0]  # flip motor 1 position data
        motor_pos_data_raw[:, 2] = -motor_pos_data_raw[:, 2]  # flip motor 3 position data
        motor_pos_data_raw[:, 4] = -motor_pos_data_raw[:, 4]  # flip motor 5 position data
        motor_pos_data_offset = 45.619998931884766 * np.ones(self.u_dim) - np.array(
            [-134.5708770751953, 224.694717407226562, -136.23983764648438, 226.012313842773438, -133.8681640625, 226.275833129882812])
        motor_pos_data_raw += motor_pos_data_offset

        def interpolate(data_raw, time_raw, time_new, kind):
            data_intp = np.zeros((len(time_new), data_raw.shape[1]))
            for col in range(data_raw.shape[1]):
                intp_function = interp1d(time_raw, data_raw[:, col], kind=kind)
                data_intp[:, col] = intp_function(time_new)
            return data_intp

        # Interpolate timestamps
        first_timestamp = max(p_time_raw[0], p_filtered_time_raw[0], v_time_raw[0], v_filtered_time_raw[0], u_time_raw[0], motor_pos_time_raw[0]) \
            if self.use_vel_est else max(p_time_raw[0], p_filtered_time_raw[0], u_time_raw[0], motor_pos_time_raw[0])
        last_timestamp = min(p_time_raw[-1], p_filtered_time_raw[-1], v_time_raw[-1], v_filtered_time_raw[-1], u_time_raw[-1], motor_pos_time_raw[-1]) \
            if self.use_vel_est else min(p_time_raw[-1], p_filtered_time_raw[-1], u_time_raw[-1], motor_pos_time_raw[-1])
        t_new = np.arange(first_timestamp, last_timestamp, self.dt)

        # Interpolate raw data
        p_data_intp = interpolate(p_data_raw, p_time_raw, t_new, 'linear')
        v_data_intp = interpolate(v_data_raw, v_time_raw, t_new, 'linear')
        p_filtered_data_intp = interpolate(p_filtered_data_raw, p_filtered_time_raw, t_new, 'linear')
        v_filtered_data_intp = interpolate(v_filtered_data_raw, v_filtered_time_raw, t_new, 'linear')
        u_data_intp = interpolate(u_data_raw, u_time_raw, t_new, 'linear')
        motor_pos_data_intp = interpolate(motor_pos_data_raw, motor_pos_time_raw, t_new, 'linear')

        def moving_average_filter(data_raw, maf_size):
            kernel = np.ones(maf_size) / maf_size
            data_filtered = np.zeros_like(data_raw)
            for col in range(data_raw.shape[1]):
                data_filtered[:, col] = convolve1d(data_raw[:, col], kernel, mode='reflect')
            return data_filtered

        # Apply a moving average filter to interpolated data
        p_data_intp_filtered = moving_average_filter(p_data_intp, self.p_maf_size)
        v_data_intp_filtered = moving_average_filter(v_data_intp, self.v_maf_size)
        u_data_intp_filtered = moving_average_filter(u_data_intp, self.u_maf_size)

        # Plot
        if self.do_plot:
            start_idx = 0
            traj_len = 100

            # Position
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            axs.plot(p_data_intp[start_idx:start_idx + traj_len, 0].flatten(), p_data_intp[start_idx:start_idx + traj_len, 1].flatten(),
                     label='/ball_pose intp')
            axs.plot(p_data_intp_filtered[start_idx:start_idx + traj_len, 0].flatten(), p_data_intp_filtered[start_idx:start_idx + traj_len, 1].flatten(),
                     label='/ball_pose intp & filter, MAF size = ' + str(self.p_maf_size * self.dt * 1000) + ' ms')
            axs.plot(p_filtered_data_intp[start_idx:start_idx + traj_len, 0].flatten(), p_filtered_data_intp[start_idx:start_idx + traj_len, 1].flatten(),
                     label='/filtered_ball_pose intp')
            axs.plot(p_data_intp[start_idx, 0], p_data_intp[start_idx, 1], 'r*', markersize=12, label='Starting point')
            axs.set_aspect('equal')
            axs.set_xlabel('X position')
            axs.set_ylabel('Y position')
            axs.legend()
            fig.suptitle(reader.path.name + ', Position Data Example', fontsize=20)

            # Velocity
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            for n in range(2):
                axs[n].plot(t_new[start_idx:start_idx + traj_len] - t_new[start_idx], v_data_intp[start_idx:start_idx + traj_len, n].flatten(), color='gray',
                            label='/ball_pose intp')
                axs[n].plot(t_new[start_idx:start_idx + traj_len] - t_new[start_idx], v_data_intp_filtered[start_idx:start_idx + traj_len, n].flatten(), color='blue',
                            label='/ball_pose intp & filter, MAF size = ' + str(self.v_maf_size * self.dt * 1000) + ' ms')
                axs[n].plot(t_new[start_idx:start_idx + traj_len] - t_new[start_idx], v_filtered_data_intp[start_idx:start_idx + traj_len, n].flatten(), color='red',
                            label='/filtered_ball_pose intp')
                axs[n].legend()
            axs[0].set_ylabel('X velocity')
            axs[1].set_ylabel('Y velocity')
            fig.suptitle(reader.path.name + ', Velocity Estimation Example, MAF Size = ' + str(self.v_maf_size), fontsize=20)

            # Control
            fig, axs = plt.subplots(6, 1, figsize=(20, 15))
            for n in range(6):
                axs[n].plot(t_new[start_idx:start_idx + traj_len] - t_new[start_idx], u_data_intp[start_idx:start_idx + traj_len, n],
                            label='/servo_angles intp')
                axs[n].plot(t_new[start_idx:start_idx + traj_len] - t_new[start_idx], u_data_intp_filtered[start_idx:start_idx + traj_len, n], '--',
                            label='/servo_angles intp & filter, MAF size = ' + str(self.u_maf_size * self.dt * 1000) + ' ms')
                axs[n].plot(t_new[start_idx:start_idx + traj_len] - t_new[start_idx], motor_pos_data_intp[start_idx:start_idx + traj_len, n],
                            label='/motor_positions intp')
            fig.suptitle(reader.path.name + ', Control Data Example', fontsize=20)
            axs[5].set_xlabel('Time(s)', fontsize=20)
            axs[5].legend()

        self.do_plot = False  # only plot data for the first bag

        # Generate dataset
        for k in range(max(self.x_delay, self.u_delay), t_new.shape[0] - 1):
            # State
            p = p_data_intp_filtered[k, :]
            p_next = p_data_intp_filtered[k + 1, :]
            q_p = p_data_intp_filtered[k - self.x_delay: k + 1, :]
            q_p_next = p_data_intp_filtered[k - self.x_delay + 1: k + 2, :]
            v = v_filtered_data_intp[k, :]
            v_next = v_filtered_data_intp[k + 1, :]
            q_v = v_filtered_data_intp[k - self.x_delay: k + 1, :]
            q_v_next = v_filtered_data_intp[k - self.x_delay + 1: k + 2, :]
            if self.use_vel_est:
                x = np.hstack([p, v])
                x_next = np.hstack([p_next, v_next])
                q = np.hstack([q_p, q_v]).flatten()
                q_next = np.hstack([q_p_next, q_v_next]).flatten()
            else:
                x = p
                x_next = p_next
                q = q_p.flatten()
                q_next = q_p_next.flatten()

            # Control and motor positions
            u = u_data_intp[k, :]
            if self.use_u_traj:
                w = u_data_intp[k - self.u_delay: k + 1, :].flatten()
                s = motor_pos_data_intp[k - self.u_delay: k + 1, :].flatten()
            else:
                w = u_data_intp[k - self.u_delay, :]
                s = motor_pos_data_intp[k - self.u_delay, :]

            # Check
            assert (np.shape(q)[0] == np.shape(q_next)[0])
            assert (np.shape(q)[0] == self.x_dim * (self.x_delay + 1))
            assert (np.array_equal(q[self.x_dim:], q_next[:-self.x_dim]))
            assert (np.array_equal(q[-self.x_dim:], x))
            assert (np.shape(w)[0] == self.u_dim * (self.u_delay + 1))
            assert (np.array_equal(w[-self.u_dim:], u))

            self.dataset['q'].append(q)
            self.dataset['q_next'].append(q_next)
            self.dataset['x'].append(x)
            self.dataset['x_next'].append(x_next)
            self.dataset['u'].append(u)
            self.dataset['w'].append(w)
            self.dataset['s'].append(s)

    def data_cutter(self):
        # Find the indices of the points have x>0 and y>0
        x_data = np.array(self.dataset['x'])

        condition = (x_data[:, 0] > 0) & (x_data[:, 1] > 0)
        idx = np.where(condition)[0]

        # Cut the data
        self.dataset['x'] = np.delete(np.array(self.dataset['x']), idx, axis=0)
        self.dataset['x_next'] = np.delete(np.array(self.dataset['x_next']), idx, axis=0)
        self.dataset['u'] = np.delete(np.array(self.dataset['u']), idx, axis=0)

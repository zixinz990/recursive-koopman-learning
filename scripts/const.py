from process_ball_pos_data import BallPosDataProcessor

PKG_PATH = '/home/stewart/Documents/Dev/koopman_ws/src/active-koopman-cpp'
rosbag_train_name_list = [PKG_PATH + '/bags/rosbag2_2024_11_04-13_53_53']

# rosbag_name_list, dt, x_delay, u_delay,
# p_maf_size, v_maf_size, u_maf_size,
# dt should be set to 10.0 ms
data_processor = BallPosDataProcessor(rosbag_train_name_list, 10.0 / 1000.0, 0, 0,
                                      1, 1, 1,
                                      use_vel_est=True, use_u_traj=True,
                                      do_plot=False)
required_num_data = 30000

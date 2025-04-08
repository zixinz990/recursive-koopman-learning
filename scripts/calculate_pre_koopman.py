from scipy.interpolate import interp1d
from scipy.linalg import logm
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from koopman import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from const import *


for name in rosbag_train_name_list:
    print(name)
data_processor.generate_dataset(required_num_data)
# data_processor.data_cutter()
num_data = len(data_processor.dataset['x'])
print('Number of data: ' + str(num_data))

obs_fun = poly_obs

# Plot the data
# fig, axs = plt.subplots(1, 1, figsize=(10, 10))
# x_data = np.array(data_processor.dataset['x'])
# axs.scatter(x_data[:, 0], x_data[:, 1])
# plt.show()

# EDMD
Y0 = []
Y1 = []
for i in range(num_data):
    x0 = data_processor.dataset['x'][i] * 100.0 # convert to cm
    u0 = data_processor.dataset['u'][i]
    x1 = data_processor.dataset['x_next'][i] * 100.0 # convert to cm
    Y0.append(np.hstack([obs_fun(x0), u0]))
    Y1.append(np.hstack([obs_fun(x1), u0]))
Y0 = np.array(Y0).T
Y1 = np.array(Y1).T

assert(Y0.shape[0] == NUM_STATE_OBS + NUM_ACTION_OBS)
assert(Y1.shape[0] == NUM_STATE_OBS + NUM_ACTION_OBS)

# Check the rank of Y0
print("Dimension of Y0: " + str(Y0.shape))
print("Rank of Y0*Y0^T: " + str(np.linalg.matrix_rank(Y0 @ Y0.T)))

# Check the inverse of Y0*Y0^T
Y0_inv = np.linalg.pinv(Y0)
err = np.linalg.norm(Y0 @ Y0_inv - np.eye(Y0.shape[0]))
print("Error of pseudo-inverse: " + str(err))

Q = Y1 @ Y0.T
P = np.linalg.pinv(Y0 @ Y0.T)
K_dt = Q @ P
K_ct = np.real(logm(K_dt, disp=False)[0] / data_processor.dt)
K_z = K_ct[0:NUM_STATE_OBS, 0:NUM_STATE_OBS]
K_u = K_ct[0:NUM_STATE_OBS, NUM_STATE_OBS:NUM_OBS]

# Remove super small values in matrices
# Y0[np.abs(Y0) < 1e-9] = 0.0
# Y1[np.abs(Y1) < 1e-9] = 0.0
# Q[np.abs(Q) < 1e-9] = 0.0
# P[np.abs(P) < 1e-9] = 0.0
# K_dt[np.abs(K_dt) < 1e-9] = 0.0
# K_z[np.abs(K_z) < 1e-9] = 0.0
# K_u[np.abs(K_u) < 1e-9] = 0.0

# Saving matrices as CSV files
Y0_df = pd.DataFrame(Y0)
Y1_df = pd.DataFrame(Y1)
Q_df = pd.DataFrame(Q)
P_df = pd.DataFrame(P)
K_dt_df = pd.DataFrame(K_dt)
K_z_df = pd.DataFrame(K_z)
K_u_df = pd.DataFrame(K_u)

Y0_df.to_csv(PKG_PATH + '/init_koopman/Y0.csv', header=False, index=False)
Y1_df.to_csv(PKG_PATH + '/init_koopman/Y1.csv', header=False, index=False)
Q_df.to_csv(PKG_PATH + '/init_koopman/Q.csv', header=False, index=False)
P_df.to_csv(PKG_PATH + '/init_koopman/P.csv', header=False, index=False)
K_dt_df.to_csv(PKG_PATH + '/init_koopman/K_dt.csv', header=False, index=False)
K_z_df.to_csv(PKG_PATH + '/init_koopman/K_z.csv', header=False, index=False)
K_u_df.to_csv(PKG_PATH + '/init_koopman/K_u.csv', header=False, index=False)

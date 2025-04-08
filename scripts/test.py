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

obs_fun = gaussian_rbf

# Construct Y0 and Y1
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

# Remove super small values in matrices
Y0[np.abs(Y0) < 1e-9] = 0.0
Y1[np.abs(Y1) < 1e-9] = 0.0

# Check the rank of Y0
print("Dimension of Y0: " + str(Y0.shape))
print("Rank of Y0*Y0^T: " + str(np.linalg.matrix_rank(Y0 @ Y0.T)))

# Check the inverse of Y0*Y0^T
Y0_inv = np.linalg.pinv(Y0)
err = np.linalg.norm(Y0 @ Y0_inv - np.eye(Y0.shape[0]))
print("Error of pseudo-inverse: " + str(err))

# QR decomposition of Y0^T
Q, R = np.linalg.qr(Y0.T)
R_row_norms = np.linalg.norm(R, axis=1)
R_row_norms_sort = np.sort(R_row_norms)
print(R_row_norms_sort)

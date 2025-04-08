from math import sqrt, floor, ceil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from process_ball_pos_data import BallPosDataProcessor
import matplotlib.pyplot as plt
from const import *


NUM_STATE_OBS = 28  # Number of state observables (poly_obs)
# NUM_STATE_OBS = 32  # Number of state observables (fourier_obs)
# NUM_STATE_OBS = 121  # Number of state observables
NUM_ACTION_OBS = 6  # Number of control inputs (servo angles)
NUM_OBS = NUM_STATE_OBS + NUM_ACTION_OBS  # Total number of observables

def poly_obs(x):
    """
    Polynomial observables for the state vector [x, y, vx, vy].

    This function maps the original state vector into a higher-dimensional
    space using polynomial terms.

    Parameters:
    - x: numpy array of shape (4,), the state vector [x, y, vx, vy]

    Returns:
    - numpy array of shape (NUM_STATE_OBS,), the transformed state vector in
      observable space
    """
    x1, x2, x3, x4 = x  # Unpack state variables: position and velocity
    return np.array([x1, x2, x3, x4,
                     x1 * x1, x2 * x2, x3 * x3, x4 * x4,
                     x1 * x2, x1 * x3, x1 * x4,
                     x2 * x3, x2 * x4, x3 * x4,
                     x1 * x1 * x1,
                     x2 * x2 * x2,
                     x3 * x3 * x3,
                     x4 * x4 * x4,
                     x1 * x1 * x2, x1 * x1 * x3, x1 * x1 * x4,
                     x2 * x2 * x3, x2 * x2 * x4,
                     x3 * x3 * x4,
                     x1 * x2 * x3, x1 * x2 * x4,
                     x1 * x3 * x4, x2 * x3 * x4])

def fourier_obs(x):
    x1, x2, x3, x4 = x
    return np.array([x1, x2, x3, x4,
                     1, 1, 1, 1,
                     np.sin(x1), np.sin(x2), np.sin(x3), np.sin(x4),
                     np.sin(2*x1), np.sin(2*x2), np.sin(2*x3), np.sin(2*x4),
                     np.sin(3*x1), np.sin(3*x2), np.sin(3*x3), np.sin(3*x4),
                     np.sin(4*x1), np.sin(4*x2), np.sin(4*x3), np.sin(4*x4),
                     np.sin(5*x1), np.sin(5*x2), np.sin(5*x3), np.sin(5*x4),
                     np.sin(6*x1), np.sin(6*x2), np.sin(6*x3), np.sin(6*x4)])

# rbf_centers = pd.read_csv(PKG_PATH + '/init_koopman/rbf_centers.csv', header=None)
# rbf_centers = rbf_centers.to_numpy()
# rbf_centers = rbf_centers * 100.0  # Convert to centimeters
# num_centers = rbf_centers.shape[0]
# assert(num_centers == NUM_STATE_OBS - 4)
# eps = 0.001

# def gaussian_rbf(x):
#     rbf = np.zeros(num_centers + 4)
#     rbf[0:4] = x
#     for i in range(num_centers):
#         d = x - rbf_centers[i]
#         rbf[i + 4] = np.exp(-eps * np.dot(d, d))
#     return rbf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
from const import *


def hexagon_grid_vertices(S, s, r_max=0.19):
    vertices = []

    delta_y = s * sqrt(3) / 2  # vertical distance between rows
    y_max = S * sqrt(3) / 2  # max y-coordinate
    j_max = int(ceil(y_max / delta_y))  # max row index

    for j in range(-j_max, j_max + 1):
        y = delta_y * j
        y_abs = abs(y)
        x_limit = S - (2 / sqrt(3)) * y_abs

        if x_limit < 0:
            continue  # skip rows outside the hexagon

        if j % 2 == 0:
            # Even row
            imin = int(ceil((-x_limit) / s))
            imax = int(floor(x_limit / s))
            for i in range(imin, imax + 1):
                x = s * i
                if (np.abs(y) + np.abs(x) / np.sqrt(3.) < r_max) and (np.abs(x) < r_max):
                    vertices.append(np.array([x, y]))
        else:
            # Odd row
            imin = int(ceil((-x_limit) / s - 0.5))
            imax = int(floor(x_limit / s - 0.5))
            for i in range(imin, imax + 1):
                x = s * (i + 0.5)
                if (np.abs(y) + np.abs(x) / np.sqrt(3.) < r_max) and (np.abs(x) < r_max):
                    vertices.append(np.array([x, y]))
    
    return vertices

def initialize_centroids(data, k, fixed_center_dim=None, fixed_center_value=None):
    n_samples, n_features = data.shape
    centroids = []

    # Randomly select the first centroid from the data points
    first_centroid_idx = np.random.randint(0, n_samples)
    centroids.append(data[first_centroid_idx])

    for _ in range(1, k):
        # Compute the distance of each point to the nearest centroid
        distances = np.min([np.linalg.norm(data - c, axis=1)**2 for c in centroids], axis=0)

        # Choose the next centroid probabilistically
        probabilities = distances / np.sum(distances)
        next_centroid_idx = np.random.choice(range(n_samples), p=probabilities)
        centroids.append(data[next_centroid_idx])
    
    # Fix part of the dimensions of the centroids
    centroids = np.array(centroids)
    if fixed_center_dim is not None and fixed_center_value is not None:
        for i in fixed_center_dim:
            centroids[:, i] = fixed_center_value[:, i]
    
    return centroids

def kmeans_plus_plus(data, k, max_iters=100, tol=1e-4, fixed_center_dim=None, fixed_center_value=None):
    # Initialize centroids using KMeans++
    centroids = initialize_centroids(data, k, fixed_center_dim, fixed_center_value)
    print("Centroids initialized")

    for iter in range(max_iters):
        print(f"Iteration {iter + 1}")

        # Assign points to the nearest centroid
        distances = np.array([np.linalg.norm(data - c, axis=1) for c in centroids])
        labels = np.argmin(distances, axis=0)

        # Update centroids as the mean of assigned points
        new_centroids = []
        for i in range(k):
            if np.any(labels == i):
                # Calculate the mean value of points assigned to the cluster
                cluster_mean = data[labels == i].mean(axis=0)
            else:
                # Retain the previous centroid if the cluster is empty
                cluster_mean = centroids[i]
            new_centroids.append(cluster_mean)
        
        # Convert the list of centroids back to a numpy array
        new_centroids = np.array(new_centroids)

        # Fix part of the dimensions of the centroids
        if fixed_center_dim is not None and fixed_center_value is not None:
            for i in fixed_center_dim:
                new_centroids[:, i] = fixed_center_value[:, i]

        # Check for convergence (centroid changes below tolerance)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids

    return centroids, labels

if __name__ == "__main__":
    np.random.seed(42)

    # Read data from the ROS bag
    data_processor.generate_dataset(required_num_data)
    x_data = data_processor.dataset['x'] # shape: (num_data, 4)
    p_data = x_data[:, :2]
    v_data = x_data[:, 2:]
    print("Complete reading data from the ROS bag")

    # Define the centers of positions
    fixed_center_dim = [0, 1]
    fixed_center_value = np.array(hexagon_grid_vertices(0.5, 0.035, r_max=0.22)) # positions
    
    # Add a small shift in y axis
    fixed_center_value[:, 1] += 0.02

    # Delete points (|x| > 0.2)
    fixed_center_value = np.array([p for p in fixed_center_value if np.abs(p[0]) < 0.2])

    print("Complete defining the centers of positions")

    # KMeans++
    k = np.shape(fixed_center_value)[0]
    centroids, labels = kmeans_plus_plus(x_data, k, fixed_center_dim=fixed_center_dim, fixed_center_value=fixed_center_value)
    print("Complete KMeans++ clustering")
    print("Number of clusters: " + str(k))

    # Save the centroids to a CSV file
    # Unit: meter
    centroids_df = pd.DataFrame(centroids)
    centroids_df.to_csv(PKG_PATH + '/init_koopman/rbf_centers.csv', header=False, index=False)
    print("The cenetrs of RBFs are saved in " + PKG_PATH + '/init_koopman/rbf_centers.csv')

    # Figure 1: Position space
    plt.figure(1)
    plt.scatter(p_data[:, 0], p_data[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Figure 2: Velocity space
    plt.figure(2)
    plt.scatter(v_data[:, 0], v_data[:, 1])
    plt.scatter(centroids[:, 2], centroids[:, 3])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

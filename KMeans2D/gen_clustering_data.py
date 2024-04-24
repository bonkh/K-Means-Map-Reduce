import numpy as np
from sklearn.datasets import make_blobs

# Save data points to a text file
def save_data_to_file(data, filename):
    np.savetxt(filename, data, fmt='%.2f')

# Generate data points
X, _ = make_blobs(n_samples=300, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)


# X = X.round(2)
X_scaled = X - X.min(axis=0)
X_scaled /= X_scaled.max(axis=0)
X_scaled *= 100


save_data_to_file(X_scaled, 'data_points.txt')

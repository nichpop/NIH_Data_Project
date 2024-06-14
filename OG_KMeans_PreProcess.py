import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

data = np.load('HF-448_V5B_1.h5_3.npy')

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def preprocess_spectrum(spectrum):
    # Calculate the baseline
    baseline = baseline_als(spectrum, 10000, 0.01)

    # Subtract the baseline
    corrected_spectrum = spectrum - baseline

    # Apply Savitzky-Golay filtering
    filtered_spectrum = savgol_filter(corrected_spectrum, 11, 3)

    # Normalize the spectrum
    norm = np.linalg.norm(filtered_spectrum)
    normalized_spectrum = filtered_spectrum / norm
    return normalized_spectrum

# data is a 3D array with shape (X, Y, 1738)
preprocessed_data = np.empty_like(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        preprocessed_data[i, j, :] = preprocess_spectrum(data[i, j, :])

# Now run k-means on preprocessed_data
# flatten data
num_x, num_y, num_spectral_points = data.shape
flattened_data = preprocessed_data.reshape(-1, num_spectral_points)

# k-means clustering
optimal_clusters = 3
kmeans = KMeans(n_clusters= optimal_clusters, random_state=1)
kmeans.fit(flattened_data)

# get the cluster labels
cluster_labels = kmeans.labels_

#unflatten data and reshape clusters
cluster_labels_reshaped = cluster_labels.reshape(num_x, num_y)

# heatmap to visualize the clusters
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_labels_reshaped, cmap='viridis')
plt.title('Cluster Map of Raman Spectra Data')
plt.xlabel('Y Coordinate')
plt.ylabel('X Coordinate')
plt.show()

# cluster analysis
# centroids of the clusters
cluster_centers = kmeans.cluster_centers_

# Analyze the size of each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

# plot the cluster centers
plt.figure(figsize=(12, 8))
for i, center in enumerate(cluster_centers):
    plt.plot(center, label=f'Cluster {i}')
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.title('Cluster Centers (Spectral Profiles)')
plt.legend()
plt.show()
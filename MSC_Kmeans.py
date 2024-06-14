import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# Load the data
data = np.load('HF-448_V5B_1.h5_3.npy')

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def preprocess_spectrum(spectrum):
    # Remove the specified ranges
    spectrum = np.concatenate((spectrum[104:907], spectrum[1374:]))
    
    # Calculate the baseline
    baseline = baseline_als(spectrum, 10000, 0.01)
    
    # Subtract the baseline
    corrected_spectrum = spectrum - baseline
    
    # Apply Savitzky-Golay filtering
    filtered_spectrum = savgol_filter(corrected_spectrum, 11, 3)
    
    return filtered_spectrum

def msc(input_data):
    # Mean spectrum
    mean_spectrum = np.mean(input_data, axis=0)
    
    # Initialize the corrected data matrix
    corrected_data = np.zeros_like(input_data)
    
    # Loop over each spectrum
    for i in range(input_data.shape[0]):
        # Perform linear regression between the spectrum and the mean spectrum
        spectrum = input_data[i, :]
        slope, intercept = np.polyfit(mean_spectrum, spectrum, 1)
        
        # Correct the spectrum
        corrected_spectrum = (spectrum - intercept) / slope
        corrected_data[i, :] = corrected_spectrum
    
    return corrected_data

# Update shape information based on the cut ranges
num_x, num_y, _ = data.shape
num_spectral_points = data.shape[2] - (104 + (1374 - 907))

# Initialize preprocessed data array with the new shape
preprocessed_data = np.empty((num_x, num_y, num_spectral_points))

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        preprocessed_data[i, j, :] = preprocess_spectrum(data[i, j, :])

# Flatten data for MSC normalization
flattened_preprocessed_data = preprocessed_data.reshape(-1, num_spectral_points)

# Apply MSC normalization
msc_corrected_data = msc(flattened_preprocessed_data)

# Normalize each spectrum to sum to 1
msc_corrected_data = msc_corrected_data / np.sum(msc_corrected_data, axis=1, keepdims=True)

# Reshape back to original shape
msc_corrected_data = msc_corrected_data.reshape(num_x, num_y, num_spectral_points)

# Now run k-means on msc_corrected_data
# Flatten data
flattened_data = msc_corrected_data.reshape(-1, num_spectral_points)

# K-means clustering
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=1)
kmeans.fit(flattened_data)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Unflatten data and reshape clusters
cluster_labels_reshaped = cluster_labels.reshape(num_x, num_y)

# Heatmap to visualize the clusters
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_labels_reshaped, cmap='viridis')
plt.title('Cluster Map of Raman Spectra Data')
plt.xlabel('Y Coordinate')
plt.ylabel('X Coordinate')
plt.show()

# Cluster analysis
# Centroids of the clusters
cluster_centers = kmeans.cluster_centers_

# Analyze the size of each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

# Plot the cluster centers
plt.figure(figsize=(12, 8))
for i, center in enumerate(cluster_centers):
    plt.plot(center, label=f'Cluster {i}')
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.title('Cluster Centers (Spectral Profiles)')
plt.legend()
plt.show()

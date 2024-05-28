import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load, flatten, and categorize data
data = np.load('HF-448_V5B_1.h5_3.npy')
num_x, num_y, num_spectral_points = data.shape
flattened_data = data.reshape(-1, num_spectral_points)

# Normalize the spectral data
scaler = StandardScaler()
flattened_data = scaler.fit_transform(flattened_data)

# k-means clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# fit model to the data
kmeans.fit(flattened_data)

# get the cluster labels
cluster_labels = kmeans.labels_
print("number of clusters found:", len(np.unique(cluster_labels)))

#unflatten data and reshape clusters
cluster_labels_reshaped = cluster_labels.reshape(num_x, num_y)
print("cluster labels reshaped to original dims:", cluster_labels_reshaped.shape)

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

# Reverse the scaling to original data space
cluster_centers_original = scaler.inverse_transform(cluster_centers)

# cluster centers
print("cluster centers (Centroids):")
print(cluster_centers_original)

# Analyze the size of each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))
print("cluster sizes:")
print(cluster_sizes)

# plot the cluster centers
plt.figure(figsize=(12, 8))
for i, center in enumerate(cluster_centers_original):
    plt.plot(center, label=f'Cluster {i}')
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.title('Cluster Centers (Spectral Profiles)')
plt.legend()
plt.show()
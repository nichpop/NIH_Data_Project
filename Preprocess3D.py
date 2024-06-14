import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

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

# Define the SNV function
def snv(input_data):
    # Subtract the mean and divide by the standard deviation
    snv_corrected_data = (input_data - np.mean(input_data, axis=1, keepdims=True)) / np.std(input_data, axis=1, keepdims=True)
    
    return snv_corrected_data

# Update shape information based on the cut ranges
num_x, num_y, _ = data.shape
num_spectral_points = data.shape[2] - (104 + (1374 - 907))

# Initialize preprocessed data array with the new shape
preprocessed_data = np.empty((num_x, num_y, num_spectral_points))

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        preprocessed_data[i, j, :] = preprocess_spectrum(data[i, j, :])

# Flatten data for SNV normalization
flattened_preprocessed_data = preprocessed_data.reshape(-1, num_spectral_points)

# Apply SNV normalization
snv_corrected_data = snv(flattened_preprocessed_data)

# Normalize each spectrum to sum to 1
snv_corrected_data = snv_corrected_data / np.sum(snv_corrected_data, axis=1, keepdims=True)

# Reshape back to original shape
snv_corrected_data = snv_corrected_data.reshape(num_x, num_y, num_spectral_points)

# Remove specified zones from raw data
data = np.concatenate((data[:, :, 104:907], data[:, :, 1374:]), axis=2)

# 3D Surface Plot for Raw Data
spectral_point_raw = 448  # Adjust the point to align with preprocessed data
intensity_values_raw = data[:, :, spectral_point_raw]
x_raw = np.arange(intensity_values_raw.shape[0])
y_raw = np.arange(intensity_values_raw.shape[1])
X_raw, Y_raw = np.meshgrid(x_raw, y_raw)
Z_raw = intensity_values_raw

fig_raw = plt.figure(figsize=(12, 8))
ax_raw = fig_raw.add_subplot(111, projection='3d')
ax_raw.plot_surface(X_raw, Y_raw, Z_raw, cmap='viridis')
ax_raw.set_title(f'3D Surface Plot of Intensity at Spectral Point {spectral_point_raw} (Raw Data)')
ax_raw.set_xlabel('X Coordinate')
ax_raw.set_ylabel('Y Coordinate')
ax_raw.set_zlabel('Intensity')
plt.show()

# 3D Surface Plot for msc Data
spectral_point_preprocessed = msc_corrected_data.shape[2] // 2
intensity_values_preprocessed = msc_corrected_data[:, :, spectral_point_preprocessed]
x_preprocessed = np.arange(intensity_values_preprocessed.shape[0])
y_preprocessed = np.arange(intensity_values_preprocessed.shape[1])
X_preprocessed, Y_preprocessed = np.meshgrid(x_preprocessed, y_preprocessed)
Z_preprocessed = intensity_values_preprocessed

fig_preprocessed = plt.figure(figsize=(12, 8))
ax_preprocessed = fig_preprocessed.add_subplot(111, projection='3d')
ax_preprocessed.plot_surface(X_preprocessed, Y_preprocessed, Z_preprocessed, cmap='viridis')
ax_preprocessed.set_title(f'3D Surface Plot of Intensity at Spectral Point {spectral_point_preprocessed} (Preprocessed Data)')
ax_preprocessed.set_xlabel('X Coordinate')
ax_preprocessed.set_ylabel('Y Coordinate')
ax_preprocessed.set_zlabel('Intensity')
plt.show()

# 3D Surface Plot for svn Data

spectral_point_preprocessed = snv_corrected_data.shape[2] // 2
intensity_values_preprocessed = snv_corrected_data[:, :, spectral_point_preprocessed]
x_preprocessed = np.arange(intensity_values_preprocessed.shape[0])
y_preprocessed = np.arange(intensity_values_preprocessed.shape[1])
X_preprocessed, Y_preprocessed = np.meshgrid(x_preprocessed, y_preprocessed)
Z_preprocessed = intensity_values_preprocessed

fig_preprocessed = plt.figure(figsize=(12, 8))
ax_preprocessed = fig_preprocessed.add_subplot(111, projection='3d')
ax_preprocessed.plot_surface(X_preprocessed, Y_preprocessed, Z_preprocessed, cmap='viridis')
ax_preprocessed.set_title(f'3D Surface Plot of Intensity at Spectral Point {spectral_point_preprocessed} (Preprocessed Data)')
ax_preprocessed.set_xlabel('X Coordinate')
ax_preprocessed.set_ylabel('Y Coordinate')
ax_preprocessed.set_zlabel('Intensity')
plt.show()

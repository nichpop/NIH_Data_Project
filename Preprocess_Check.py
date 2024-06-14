import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data = np.load('HF-448_V5B_1.h5_3.npy')

#baseline correction helper
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

# Choose a spectrum to plot
spectrum = data[30, 30, :]

# Calculate the baseline
spectrum = np.concatenate((spectrum[104:907], spectrum[1374:]))
baseline = baseline_als(spectrum, 10000, 0.01)

# Plot the original spectrum and the baseline
plt.figure(figsize=(10, 6))
plt.plot(spectrum, label='Original')
plt.plot(baseline, label='Baseline')
plt.legend()
plt.show()

# Plot the corrected spectrum
plt.figure(figsize=(10, 6))
plt.plot(spectrum - baseline, label='Corrected')
plt.legend()
plt.show()



# Apply Savitzky-Golay filtering
filtered_spectrum = savgol_filter(spectrum, 11, 3)

# Plot the original and filtered spectra
plt.figure(figsize=(10, 6))
plt.plot(spectrum, label='Original')
plt.plot(filtered_spectrum, label='Filtered')
plt.legend()
plt.show()

# Initialize a new array to hold the normalized spectra
normalized_data = np.empty_like(data)

# Iterate over the X,Y coordinates
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # Select a spectrum
        spectrum = data[i, j, :]

        # Normalize the spectrum
        normalized_spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

        # Store the normalized spectrum
        normalized_data[i, j, :] = normalized_spectrum

# Select a spectrum to plot
normalized_spectrum = normalized_data[30, 30, :]


# Plot the original and normalized spectra
plt.figure(figsize=(10, 6))
plt.plot(spectrum, label='Original')
plt.plot(normalized_spectrum, label='Normalized')
plt.legend()
plt.show()




# Select a spectrum
spectrum = data[30, 30, :]

# Calculate the baseline
spectrum = np.concatenate((spectrum[104:907], spectrum[1374:]))
baseline = baseline_als(spectrum, 10000, 0.01)

# Subtract the baseline
corrected_spectrum = spectrum - baseline

# Apply Savitzky-Golay filtering
filtered_spectrum = savgol_filter(corrected_spectrum, 11, 3)

# Normalize the spectrum
norm = np.linalg.norm(filtered_spectrum)
if norm == 0:
    print("Warning: Norm of spectrum is zero")
    normalized_spectrum = filtered_spectrum
else:
    normalized_spectrum = filtered_spectrum / norm


# Plot the original spectrum, the baseline, and the preprocessed spectrum
plt.figure(figsize=(10, 6))
plt.plot(spectrum, label='Original')
plt.plot(baseline, label='Baseline')
plt.plot(normalized_spectrum, label='Preprocessed')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(normalized_spectrum, label='Normalized')
plt.legend()
plt.show()
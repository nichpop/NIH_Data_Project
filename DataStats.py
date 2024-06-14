import numpy as np
import pandas as pd

data1 = np.load('HF-448_V5B_1.h5_3.npy')
data2 = np.load('HF-868_1_2.h5_4.npy')
print(data1.shape)
print(data2.shape)
data1 = np.concatenate((data1[:, :, :104], data1[:, :, 907:1374], data1[:, :, 1374:]), axis=2)
data2 = np.concatenate((data2[:, :, :104], data2[:, :, 907:1374], data2[:, :, 1374:]), axis=2)
print(data1.shape)
print(data2.shape)

# basic stats
mean1 = np.mean(data1)
median1 = np.median(data1)
std_dev1 = np.std(data1)
min_value1 = np.min(data1)
max_value1 = np.max(data1)

mean2 = np.mean(data2)
median2 = np.median(data2)
std_dev2 = np.std(data2)
min_value2 = np.min(data2)
max_value2 = np.max(data2)

# results
print(f"Mean: {mean1} , {mean2}")
print(f"Median: {median1}, {median2}")
print(f"Standard Deviation: {std_dev1}, {std_dev2}")
print(f"Minimum Value: {min_value1}, {min_value1}")
print(f"Maximum Value: {max_value1}, {max_value1}")

# save excel and flatten
# Load the .npy file
data = np.load('HF-448_V5B_1.h5_3.npy')

# Flatten the 3D array to a 2D array
flat_data = []

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            flat_data.append([i, j, k, data[i, j, k]])

# Convert the flattened data to a DataFrame
df = pd.DataFrame(flat_data, columns=['dim1', 'dim2', 'dim3', 'value'])

# Select a subset of the data that fits into an Excel sheet
# Excel sheet limit is 1,048,576 rows
max_rows = 100000
df_subset = df.iloc[:max_rows]
df_subset.to_excel('flattened_HF-448_V5B_1.h5_3.xlsx', index=False)
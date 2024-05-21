import numpy as np

data1 = np.load('HF-448_V5B_1.h5_3.npy')
data2 = np.load('HF-868_1_2.h5_4.npy')

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

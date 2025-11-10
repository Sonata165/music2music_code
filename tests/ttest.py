# import numpy as np
# from scipy.stats import ttest_ind

# # Example data
# data1 = np.random.normal(loc=50, scale=10, size=100)  # Group 1
# data2 = np.random.normal(loc=60, scale=12, size=80)   # Group 2

# # Perform Welch's T-test
# t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

# print("T-statistic:", t_stat)
# print("P-value:", p_value)

import numpy as np
from scipy.stats import ttest_ind

# Data from user's description
data1 = [5, 3, 5, 3, 2, 3, 3, 4, 3, 4, 4, 3, 5, 5, 5, 3, 4, 4, 4, 4, 2, 3, 4, 2, 4, 4, 4, 3, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 3, 3, 3, 4, 5, 5, 3, 4, 5, 2, 2, 5, 5, 3, 1, 4, 5]
data2 = [2, 1, 3, 4, 3, 3, 4, 5, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 3, 2, 1, 3, 4, 2, 3, 2, 4, 3, 4, 3, 3, 3, 3, 2, 4, 4, 2, 2, 2, 3, 3, 3, 2, 4, 2, 3, 3, 2, 3, 2, 2, 3, 3, 2, 1, 3, 4]

# Perform Welch's T-test (suitable for unequal variances and sample sizes)
t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_value)


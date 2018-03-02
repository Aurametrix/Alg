import numpy as np
print(np.__version__)

# create an array
arr = np.arange(10)
arr

# creating a boolean array
np.full((3, 3), True, dtype=bool)
# or
np.ones((3,3), dtype=bool)

# extrat odd numbers fom an array
# Input
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Solution
arr[arr % 2 == 1]
#> array([1, 3, 5, 7, 9])

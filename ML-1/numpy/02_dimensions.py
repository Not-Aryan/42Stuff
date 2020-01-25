import numpy as np

# a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# b = np.array([(1.5,2,3), (4,5,6)], dtype = float)

a = np.arange(10)
a = np.reshape(a, (2, 5))

print(a)
# print(a.ndim)
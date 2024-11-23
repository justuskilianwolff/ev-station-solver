import numpy as np

a = np.empty((0, 2))  
print(a)
b = np.append(a, np.array([[1, 2], [2, 3]]), axis=0)

print(b)

print(np.array([]).shape)
print(np.empty((0,)).shape)
import numpy as np
import matplotlib.pyplot as plt

# make array, print ndim, shape
x = np.array(12)
print(x, x.ndim, x.shape)
print(len(x.shape))

x = np.array([1, 2, 3])
print(x, x.ndim, x.shape)
print(len(x.shape))

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x, x.ndim, x.shape)
print(len(x.shape))

# using reshape
x = x.reshape((1, 6))
print(x, x.ndim, x.shape)
print(len(x.shape))

x = x.reshape(6)
print(x, x.ndim, x.shape)
print(len(x.shape))

# using matplot binary
digit = np.array([[0, 0, 8], [0, 5, 1], [100, 0, 2]])
# plt.imshow(digit, cmap=plt.cm.binary)
plt.imshow(digit)
plt.show()
#---------  -------------------------------
x = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6], [2.7, 2.8, 2.9]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[4, 4, 4], [4, 4, 4], [4, 4, 4]]])
print ("x:", x.ndim, x.shape)

print("x[1]:", x[1])
print("x[2]:", x[2])
y = x[1:3]
print ("y:", y, y.ndim, y.shape)

y = x[1:3, 0:1, 0:2]
print ("\nx[1:3, 0:1, 0:2]\n", y, y.ndim, y.shape)

rn = np.random.random((2,3,2))
print("rn:", rn)

x = np.array([[[1,1],[1,1]],[[5,2],[3,3]]])
print (x)
y = np.array([[2,2],[3,3]])
print (y)
z = np.maximum(x,y)
print("z:",z)

x = np.array([[1,2,3],[0,0,0]])
y = np.array([4,2,4])
print (x.shape)
x[1] += y[1]
print (x)

x = np.array([1,1,1])
y = np.array([[2],[2],[2]])
z = np.dot(x,y)
print (z)

l = []
l.append([1,2])
l.append([3,4])
l.append([5,6])
x = np.array(l)
print(x)
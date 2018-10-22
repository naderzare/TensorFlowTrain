import numpy as np

x = np.array(12)
print (x,x.ndim,x.shape)
print(len(x.shape))

x = np.array([1,2,3])
print (x,x.ndim,x.shape)
print(len(x.shape))

x = np.array([[1,2,3],[4,5,6]])
print (x,x.ndim,x.shape)
print(len(x.shape))

x = x.reshape((1,6))
print (x,x.ndim,x.shape)
print(len(x.shape))

x = x.reshape((6))
print (x,x.ndim,x.shape)
print(len(x.shape))

digit = np.array([[0,0,0],[0,0.5,1],[0,0,0]])
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
#----------------------------------------
x = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6], [2.7, 2.8, 2.9]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[4, 4, 4], [4, 4, 4], [4, 4, 4]]])
print (x.ndim, x.shape)

y = x[1:3]
print (y, y.ndim, y.shape)

y = x[1:3, 0:2, 0:3]
print (y, y.ndim, y.shape)

rn = np.random.random((2,3,2))
print(rn)

x = np.array([[[1,1],[1,1]],[[5,2],[3,3]]])
print (x)
y = np.array([[2,2],[3,3]])
print (y)
z = np.maximum(x,y)
print(z)

x = np.array([[1,2,3],[0,0,0]])
y = np.array([4,2,4])
print (x.shape)
x[1] += y[1]
print (x)

x = np.array([1,1,1])
y = np.array(2)
z = np.dot(x,y)
print (z)
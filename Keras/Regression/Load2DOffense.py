from keras import models
from numpy import array, random

xin = open('xout', 'r')
yin = open('yout', 'r')

x = xin.readlines()
y = yin.readlines()

X = []
Y = []
for xx in x:
    xxx = xx.split(' ')
    X.append([float(xxx[0]) / 52.5, float(xxx[1]) / 34.0])

for xx in y:
    Y.append(float(xx))

X = array(X)
Y = array(Y)

model = models.load_model('model.h5')
YY = model.predict(X)

print(Y.shape, YY.shape)
D = 0
for i in range(Y.shape[0]):
    d = abs(Y[i] - YY[i][0])
    print(Y[i], YY[i][0], d)
    D += d

print(X.shape)
x1 = [x[0] for x in X]
x2 = [x[1] for x in X]
import matplotlib.pyplot as plt
plt.clf()
YY = YY.reshape((YY.shape[0]))
plt.scatter(x1, x2, c=YY)
plt.show()

import math
X1 = []
X2 = []
Y = []
x1 = -52.5
x2 = -34
while x1 <= 52.5:
    x2 = -34
    while x2 <= 34:

        y1 = math.sqrt(pow(abs(52.5 - x1), 2) + pow(abs(0 - x2), 2))
        y = x1
        if y1 < 40:
            y += (40 - y1)
        X1.append(x1)
        X2.append(x2)
        Y.append(y)
        x2 += 0.5
    x1 += 0.5
xout = open('xout', 'w')
yout = open('yout', 'w')
for i in range(len(X1)):
    xout.write(str(X1[i]) + ' ' + str(X2[i]) + '\n')
    yout.write(str(Y[i]) + '\n')
xout.close()
yout.close()

import matplotlib.pyplot as plt
plt.clf()
plt.scatter(X1, X2, c=Y)
plt.show()
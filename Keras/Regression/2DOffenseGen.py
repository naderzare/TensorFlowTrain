def BaseEvaluator(x, y):
    dist = pow(pow(105 - x, 2) + pow(abs(35 - y), 2), 0.5)
    e = x
    if dist < 40:
        e += (40 - dist)
    return e


X = [(x / 2, y / 2) for x in range(0, 105 * 2, 1) for y in range(0, 70 * 2, 1)]
Y = [BaseEvaluator(x, y) for x, y in X]


xin = open('xout', 'w')
yin = open('yout', 'w')
for i in range(len(X)):
    xin.write(str(X[i][0] / 105) + ' ' + str(X[i][1] / 70) + '\n')
    yin.write(str(Y[i] / 145) + '\n')
xin.close()
yin.close()

import matplotlib.pyplot as plt
plt.clf()
plt.scatter([x[0] for x in X], [x[1] for x in X], c=Y)
plt.show()


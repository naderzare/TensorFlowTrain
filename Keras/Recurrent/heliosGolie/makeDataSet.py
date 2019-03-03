from keras import layers, models, activations, losses, metrics, optimizers
import matplotlib.pyplot as plt
from numpy import array, random
import random


datas = []
sequence_data = []

print('Generating Data...')
file_name = '20190227202239-HELIOS2018_23-vs-CYRUS2018_8.rcg'
f = open('logs/'+file_name, 'r')
lines = f.readlines()
mode = 'setplay'
gen = lambda l, vmax: [(l[0]+52.5)/105, (l[1]+34.0)/68.0, (l[2]+vmax)/(vmax * 2.0), (l[3]+vmax)/(vmax * 2.0)]
for l in lines:
    left_players = {}
    right_players = {}

    if l.startswith('(playmode'):
        if l.find('play_on') > 0:
            mode = 'play_on'
        else:
            mode = 'setplay'
    if mode == 'setplay' and len(sequence_data) > 0:
        datas.append(sequence_data)
        sequence_data = []
        continue
    if not l.startswith('(show') or mode == 'setplay':
        continue
    ball = [float(b) for b in l.replace(')', '').split(' ')[3:7]]
    if ball[1] > 0:
        continue
    ball = gen(ball, 3.0)
    goalie = [float(p) for p in l.split(' ')[11:15]]

    for u in range(1, 23):
        side = 'l'
        unum = u
        if u > 11:
            unum = u - 11
            side = 'r'
        start_pos = l.find('(({} {})'.format(side, unum))
        string_f = l[start_pos:].split(' ')[4:8]
        float_f = [float(p) for p in string_f]
        f = gen(float_f, 1.0)
        if side == 'l':
            left_players[unum] = f
        else:
            right_players[unum] = f

    sequence_data.append(ball + left_players[1])

DATAX = []
DATAY = []
import math
for d in datas:
    for s in range(len(d) - 8):
        DX = d[s:s+5]
        x1 = DX[-1][4] * 105.5 - 52.5
        y1 = DX[-1][5] * 68.0 - 34.0
        x2 = (d[s+5][4] + d[s+6][4] + d[s+7][4]) / 3 * 105.0 - 52.5
        y2 = (d[s + 5][5] + d[s + 6][5] + d[s + 7][5]) / 3 * 68.0 - 34.0
        y = y2 - y1
        x = x2 - x1
        an = 0
        if x == 0 and y == 0:
            an = 0
        else:
            an = math.atan2(y, x) / 3.14 * 180.0
        an = (an + 180.0) / 360.0
        DY = d[s+7][4:6]
        DATAX.append(DX)
        # DATAY.append(DY)
        DATAY.append(an)
print(len(DATAX),len(DATAY))
# print(len(DATAX[0]),len(DATAY[0]))

r = list(range(len(DATAX)))
random.shuffle(r)
X = [DATAX[x] for x in r]
Y = [DATAY[x] for x in r]

train_number = int(len(Y) * 0.9)
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:]
Y_test = Y[train_number:]

X_train = array(X_train)
X_test = array(X_test)
Y_train = array(Y_train)
Y_test = array(Y_test)


print(X_train.shape)
network = models.Sequential()
network.add(layers.LSTM(5))
# network.add(layers.Dropout(0.2, input_shape=(8,)))
network.add(layers.Dense(20, activation=activations.elu, input_shape=(X_train.shape[-1],)))
network.add(layers.Dense(10, activation=activations.elu))
network.add(layers.Dense(1, activation=activations.elu))
network.compile(optimizer=optimizers.Adam(), loss=losses.mse, metrics=[metrics.mse])
history = network.fit(X_train, Y_train, epochs=1000, batch_size=64, validation_data=(X_test, Y_test))
# test_loss, test_acc = network.evaluate(test_datas, test_labels)
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['mean_squared_error']
val_acc_values = history_dict['val_mean_squared_error']

epochs = range(len(loss_values))
plt.figure(1)
plt.subplot(211)
plt.plot(epochs, loss_values, 'r--', label='Training loss')
plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')
plt.title("train/test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
plt.plot(epochs, acc_values, 'r--', label='Training mean_squared_error')
plt.plot(epochs, val_acc_values, '--', label='Validation mean_squared_error')
plt.title("train/test acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
network.save('model.h5')

D = []
D.append([-24.53,27.90,0.28,0.10,-20.49,31.23,-0.4,-0.33])
D.append([-23.76,28.15,0.25,0.08,-20.86,30.90,-0.36,-0.31])
D.append([-23.49,28.21,0.09,0.02,-21.22,30.58,0.03,0.03])
D.append([-23.03,28.69,0.15,0.16,-21.21,31.10,0.01,0.48])
D.append([-22.54,29.26,0.16,0.19,-21.2,31.58,0.01,0.45])
D = [[(d[4] + 52.5)/105.0, (d[5]+34)/68.0, (d[6]+3.0)/6.0, (d[7]+3.0)/6.0, (d[0]+52.5)/105.0, (d[1]+34)/68.0, (d[2]+1.5)/3.0, (d[3]+1.5)/3.0] for d in D]
D = [D]
D =array(D)
a = network.predict(D)
print(a)

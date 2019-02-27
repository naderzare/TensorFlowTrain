from keras import layers, models, activations, losses, metrics, optimizers
import matplotlib.pyplot as plt
from numpy import array, random
import random

f = open('data.csv', 'r')
L = f.readlines()
X = [l.split(',')[0:3] for l in L]
X = list(map(lambda x: [float(y) for y in x], X))

Y = [l.split(',')[3] for l in L]
Y = list(map(lambda y: float(y), Y))

r = list(range(len(Y)))
random.shuffle(r)
X = [X[x] for x in r]
Y = [Y[x] for x in r]

train_number = int(len(Y) * 0.9)
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:]
Y_test = Y[train_number:]

use_recurrent = True
if use_recurrent:
    X_train = [[[t[0]], [t[1]], [t[2]]] for t in X_train]
    X_test = [[[t[0]], [t[1]], [t[2]]] for t in X_test]
X_train = array(X_train)
X_test = array(X_test)
Y_train = array(Y_train)
Y_test = array(Y_test)

network = models.Sequential()
if use_recurrent:
    network.add(layers.LSTM(3))
network.add(layers.Dense(20, activation=activations.relu, input_shape=(3,)))
network.add(layers.Dense(10, activation=activations.relu))
network.add(layers.Dense(1, activation=activations.relu))
network.compile(optimizer=optimizers.Adam(), loss=losses.mse, metrics=[metrics.mse])
history = network.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))
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





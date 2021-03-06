from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import random
from numpy import array

f = open('kick', 'r')
lines = f.readlines()

X = []
Y = []
for l in lines:
    x = l.split('|')[0].split(',')
    x = [float(v) for v in x[1:]]
    y = int(l.split('|')[1])
    x[1] = (x[1] + 180.0) / 360.0
    x[3] = (x[3] + 180.0) / 360.0
    x[5] = (x[5] + 180.0) / 360.0
    x[7] = (x[7] + 180.0) / 360.0
    x[2] = (x[2] / 30.0)
    x[6] = x[6]/3.0
    X.append(x)
    if y == -1:
        y = 0
    Y.append(y)

r = list(range(len(X)))
random.shuffle(r)
X = [X[x] for x in r]
Y = [Y[y] for y in r]

train_number = int(len(Y) * 0.9)
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:]
Y_test = Y[train_number:]

X_train = array(X_train)
X_test = array(X_test)
Y_train = array(Y_train)
Y_test = array(Y_test)

Y_train = to_categorical(Y_train)

Y_test = to_categorical(Y_test)
network = models.Sequential()
network.add(layers.Dense(20, activation='elu', input_shape=(8,)))
network.add(layers.Dense(20, activation='elu'))
network.add(layers.Dense(4, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = network.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
history_dict = history.history
print(history_dict['val_loss'])
print(history_dict['val_acc'])
print(history_dict['loss'])
print(history_dict['acc'])
epochs = range(1, len(history_dict['acc']) + 1)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.clf()
plt.figure(1)
plt.subplot(211)
plt.plot(epochs, loss_values, 'r--', label='Training loss')
plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')
plt.title("train/test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
plt.plot(epochs, acc_values, 'r--', label='Training acc')
plt.plot(epochs, val_acc_values, '--', label='Validation acc')
plt.title("train/test acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

network.save('model.h5')
# json_model = network.to_json()
# f = open("model.h5", "w")
# f.write(json_model)
# network.save_weights('weights.h5')

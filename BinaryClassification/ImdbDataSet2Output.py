from keras.datasets import imdb
import numpy as np

# include 10000 words that has max frequent in review text
# len(train_data) = 25000 means 25000 reviews
# max(len(train_data[i])) = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# review1 = 5 10 1 8
# to
# review1 = 0 1 0 0 0 1 0 0 1 0 1
def vectorize_sequences(sequences, dimension = 10000):
    res = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        res[i, sequence] = 1
    return res
def vectorize_labels(labels):
    res = np.zeros((len(labels), 2))
    for l in range(len(labels)):
        if labels[l] < 0.5:
            res[l][0] = 1
        else:
            res[l][1] = 1
    return res
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = vectorize_labels(train_labels)#np.asarray(train_labels).astype('float32')
y_test = vectorize_labels(test_labels)#np.asarray(test_labels).astype('float32')
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]

from keras import models
from keras import layers

layer_activ = ['relu', 'relu', 'softmax']
optim_loss = ['rmsprop', 'binary_crossentropy']
model = models.Sequential()
model.add(layers.Dense(16, activation=layer_activ[0], input_shape=(10000,)))
model.add(layers.Dense(16, activation=layer_activ[1]))
model.add(layers.Dense(2, activation=layer_activ[2]))
model.compile(optimizer=optim_loss[0], loss=optim_loss[1], metrics=['accuracy'])
epoch_number = 10
history = model.fit(x_train,y_train,epochs=epoch_number, batch_size=256,validation_data=(x_val,y_val))

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, epoch_number + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
fname = 'TrainingAndValidationLoss2out,[{},{},{}],optim={},loss={}'.format(layer_activ[0],layer_activ[1],layer_activ[2],optim_loss[0],optim_loss[1])
plt.title(fname)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
plt.clf()

history_dict = history.history
loss_values = history_dict['acc']
val_loss_values = history_dict['val_acc']
epochs = range(1, epoch_number + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
fname = 'TrainingAndValidationAcc2out,[{},{},{}],optim={},loss={}'.format(layer_activ[0],layer_activ[1],layer_activ[2],optim_loss[0],optim_loss[1])
plt.title(fname)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


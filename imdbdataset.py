from keras.datasets import imdb
import numpy as np

from keras import losses
from keras import backend as K
tv = K.variable(np.array([[1]]))
pv = K.variable(np.array([[1.2]]))


print(K.eval(losses.mean_squared_error(tv,pv)))
print(K.eval(losses.binary_crossentropy(tv,pv)))


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_labels[4])
exit()

def vectorize_sequences(sequences, dimension = 10000):
    res = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        res[i, sequence] = 1
    return res
print("train data dim {} shape {}".format(train_data.ndim,train_data.shape))
print("train labels dim {} shape {}".format(train_labels.ndim,train_labels.shape))
print("test data dim {} shape {}".format(test_data.ndim,test_data.shape))
print("test labels dim {} shape {}".format(test_labels.ndim,test_labels.shape))

print (len(train_data))
for i in range(10):
    print(len(train_data[i]))
    print(train_labels[i])
print('number of word:',max([max(sequence) for sequence in train_data]))

print('max review len:',max([len(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
x = word_index["go"]
print(x)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index[x])
print(reverse_word_index[1])
print(reverse_word_index[2])

decoded_review = ''.join([(reverse_word_index.get(i, '?') + ' ') for i in train_data[0]])
print (decoded_review)

x_train = vectorize_sequences(train_data)
print(x_train[0])
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

partial_x_train = x_train[:10000]
val_x_train = x_train[10000:]
partial_y_train = y_train[:10000]
val_y_train = y_train[10000:]

history = model.fit(partial_x_train,partial_y_train,
                    epochs=10, batch_size=256,validation_data=(val_x_train,val_y_train))

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 10 + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
# plt.clf()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()

predicted = model.predict(x_test[:10])

print(predicted,y_test[:10])
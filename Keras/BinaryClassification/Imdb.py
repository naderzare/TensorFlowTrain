from keras.datasets import imdb
from keras import layers, models, optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)


def vectorize_sequence(sequences, num_words=10000):
    res = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
        res[i, seq] = 1
    return res


train_data = vectorize_sequence(train_data, 1000)
train_labels = np.asarray(train_labels).astype('float32')
test_data = vectorize_sequence(test_data, 1000)
test_labels = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(20, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
history = model.fit(train_data,
                    train_labels,
                    batch_size=32,
                    epochs=5,
                    validation_data=(test_data, test_labels),
                    verbose=0)
history_dict = history.history

epochs = range(1, len(history_dict['loss']) + 1)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

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

model.save('model.h5')
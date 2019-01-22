from keras.datasets import reuters
from keras import models, layers, activations, optimizers, losses, metrics
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(100, activation=activations.relu, input_shape=(10000,)))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(70, activation=activations.relu))
model.add(layers.Dense(46, activation=activations.softmax))
model.compile(optimizer=optimizers.Adam(),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])
history = model.fit(train_data, train_labels,
                    batch_size=32,
                    epochs=20,
                    validation_data=(test_data, test_labels),
                    verbose=0)

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['categorical_accuracy']
val_acc_values = history_dict['val_categorical_accuracy']

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
plt.plot(epochs, acc_values, 'r--', label='Training acc')
plt.plot(epochs, val_acc_values, '--', label='Validation acc')
plt.title("train/test acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

model.save('model.h5')


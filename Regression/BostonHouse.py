from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

maximum = []
mean = []
for i in range(13):
    maximum.append(np.max(train_data[:][i]))
    mean.append(np.mean(train_data[:][i]))

train_data /= maximum
train_data -= 0.5
test_data /= maximum
test_data -= 0.5
# maximum = np.max(train_targets)
# train_targets /= maximum
# test_targets /= maximum

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train_data,train_targets, batch_size=50,epochs=100, validation_data=(test_data,test_targets))
print (history.history)

epoches = range(1, 101)
acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
plt.plot(epoches, acc, 'bo', label='Training acc')
plt.plot(epoches, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
print(history.history['val_mean_absolute_error'][-1] )
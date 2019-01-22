from keras.datasets import boston_housing
from keras import layers, models, activations, losses, metrics, optimizers
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
test_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data /= std

model = models.Sequential()
model.add(layers.Dense(100, activation=activations.relu, input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(1))

model.compile(optimizer=optimizers.Adam(), loss=losses.mse, metrics=[metrics.mse])

history = model.fit(train_data, train_targets, batch_size=32, epochs=100, validation_data=(test_data, test_targets))

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

model.save('model.h5')


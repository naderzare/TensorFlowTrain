from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import Input
from keras import Model
import matplotlib.pyplot as plt
from numpy import array

X = [[x/1000, y/1000] for x in range(1000) for y in range(1000)]
print(len(X))
Y1 = [(x[0] + x[1]) for x in X]
Y2 = list(map(lambda x: int(x[0] > 0.5), X))
Y2 = to_categorical(Y2)

X = array(X)
Y1 = array(Y1)
Y2 = array(Y2)

print(X.shape, Y1.shape, Y2.shape)

model_input = Input(shape=(2,))
x = layers.Dense(20, activation='relu')(model_input)
x = layers.Dense(10, activation='relu')(x)
y1 = layers.Dense(1, activation='relu')(x)
y2 = layers.Dense(2, activation='softmax')(x)
model = Model(input=model_input, output=[y1, y2])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'accuracy'])
history = model.fit(X, [Y1, Y2], epochs=2, batch_size=32)

a = [[0.8, 0.3], [0.2, 0.6]]
a = array(a)

print(model.predict(a))
exit()

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

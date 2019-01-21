from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#show one of train image
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = network.fit(train_images, train_labels, epochs=2, batch_size=32, validation_data=(test_images, test_labels))
# test_loss, test_acc = network.evaluate(test_images, test_labels)

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

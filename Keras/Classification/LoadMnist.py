from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network = models.load_model("../BinaryClassification/model.h5")

test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss, test_acc)

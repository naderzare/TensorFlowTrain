from keras.datasets import imdb
from keras import models
import numpy as np

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
network = models.load_model("model.h5")

test_loss, test_acc = network.evaluate(test_data, test_labels)
print(test_loss, test_acc)

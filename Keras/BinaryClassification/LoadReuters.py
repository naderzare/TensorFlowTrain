from keras.datasets import reuters
from keras import models
import numpy as np
from keras.utils.np_utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


test_data = vectorize_sequences(test_data)
test_labels = to_categorical(test_labels)

model = models.load_model('model.h5')
print(model.evaluate(test_data, test_labels))


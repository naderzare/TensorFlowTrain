from keras.datasets import reuters


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(max(train_labels))
print(max([max(x) for x in train_data]))

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# train_data 25000 * list, each list has number between 1 and 9999
# train_label 25000 * [0 or 1]
word_index = imdb.get_word_index()
# ['reverent': 44834, 'gangland': 22426, "'ogre'": 65029, 'prolly': 28701 ... ]
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
# reverse_word_index[1] = the
decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decode_review)

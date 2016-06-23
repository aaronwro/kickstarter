"""
Significant contributions from:
* https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py
* https://github.com/dennybritz/cnn-text-classification-tf

Based on the paper by Yoon Kim:
* http://arxiv.org/abs/1408.5882
* https://github.com/yoonkim/CNN_sentence

Relevant Copyright/License from contributions:

All contributions by François Chollet:
Copyright (c) 2015, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015, Google, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D
import pandas as pd
import re
import itertools
from collections import Counter

WINDOW_SIZE = 25

# Embedding
max_features = vocabulary_size
maxlen = WINDOW_SIZE 
embedding_size = 128
# TODO: when loading data, truncate example to 25 or wrap sentances to a shorter length.
# TODO: when loading data, eliminiate words that are infrequent, and replace them with UNK tokens, or UNK token classes
# padding uses the zero index of my embeddings
# last index of embeddings should by my UNK

# Convolution
filter_length = 2        # number of words considered at a time
nb_filter = 512          # number of convolutions to scan the input
pool_length = 2          # pool half of the filtered inputs

# LSTM
lstm_output_size = 32   

# Training
batch_size = 512          # scale this as much as the machine will allow
nb_epoch = 10             # more epochs!

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary, test=False):
	"""
	Maps sentencs and labels to vectors based on a vocabulary.
	"""
	x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
	y = np.array(labels)
	return [x, y]

def chunks(in_list, chunk_size):
    return [in_list[x:x+chunk_size] for x in xrange(0, len(in_list), chunk_size)]

def load_data_in_windows():
	"""
	Loads and preprocessed data for the Kickstarter dataset.
	Returns input vectors, labels, vocabulary, and inverse vocabulary.
	"""
	# Load and preprocess data
	positive_examples = []
	negative_examples = []
	positive_labels = []
	negative_labels = []
	df = pd.read_csv('kickstarter_assignment/kickstarter_corpus_cleaned.csv')
	print('data shape: ' + str(df.shape))
	# split test and train data
	test_projects = df[:100]
	train_projects = df[100:]

	vocabulary, vocabulary_inv = build_vocab([sentance for sentance in df['cleaned_words'].str.split()])

	funded_training_sentances = train_projects[train_projects.funded == True]['cleaned_words'].str.split()
	not_funded_training_sentances = train_projects[train_projects.funded == False]['cleaned_words'].str.split()
	positive_training_words = [word for sentance in funded_training_sentances for word in sentance]
	negative_training_words = [word for sentance in not_funded_training_sentances for word in sentance]

	print('positive_training_words[:3]: ' + str(positive_training_words[:3]))
	print('negative_training_words[:3]: ' + str(negative_training_words[:3]))

	positive_training_ints = np.array([vocabulary[word] for word in positive_training_words])
	negative_training_ints = np.array([vocabulary[word] for word in negative_training_words])

	print('positive_training_ints[:3]: ' + str(positive_training_ints[:3]))
	print('negative_training_ints[:3]: ' + str(negative_training_ints[:3]))

	positive_training_chunks = chunks(positive_training_ints, WINDOW_SIZE)
	negative_training_chunks = chunks(negative_training_ints, WINDOW_SIZE)
	print('positive training chunks: ' + str(len(positive_training_chunks)))
	print('negative training chunks: ' + str(len(negative_training_chunks)))

	print('positive_training_chunks[:3]: ' + str(positive_training_chunks[:3]))
	print('negative_training_chunks[:3]: ' + str(negative_training_chunks[:3]))

	funded_testing_sentances = test_projects[test_projects.funded == True]['cleaned_words'].str.split()
	not_funded_testing_sentances = test_projects[test_projects.funded == False]['cleaned_words'].str.split()
	positive_testing_words = [word for sentance in funded_testing_sentances for word in sentance]
	negative_testing_words = [word for sentance in not_funded_testing_sentances for word in sentance]

	positive_testing_ints = np.array([vocabulary[word] for word in positive_testing_words])
	negative_testing_ints = np.array([vocabulary[word] for word in negative_testing_words])

	positive_testing_chunks = chunks(positive_testing_ints, WINDOW_SIZE)
	negative_testing_chunks = chunks(negative_testing_ints, WINDOW_SIZE)
	print('positive testing examples: ' + str(len(positive_testing_chunks)))
	print('negative testing examples: ' + str(len(negative_testing_words)))

	x_train = np.concatenate([positive_training_chunks, negative_training_chunks])
	y_train = np.concatenate([np.ones(len(positive_training_chunks), dtype=np.int), np.zeros(len(negative_training_chunks), dtype=np.int)])

	x_test = np.concatenate([positive_testing_chunks, negative_testing_chunks])
	y_test = np.concatenate([np.ones(len(positive_testing_chunks), dtype=np.int), np.zeros(len(negative_testing_chunks), dtype=np.int)])

	return [x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv]

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
print("Loading data...")
X_train, y_train, X_test, y_test, vocabulary, vocabulary_inv = load_data_in_windows()
vocabulary_size = len(vocabulary)
print("Vocabulary Size: {:d}".format(vocabulary_size))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.8))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))                        # 1 neuron
model.add(Activation('sigmoid'))           # can change to a hard sigmoid for % funded

# could use mse metrics instead of accuracy to check for % funded
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

json_string = model.to_json()
with open('kickstarter_keras.json', 'w') as outfile:
    outfile.write(json_string)

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
model.save_weights('kickstarter_keras.h5')

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

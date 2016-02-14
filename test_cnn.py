import cPickle as pickle

import numpy as np
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD

from nn import train_neural
from cnn.layers.embeddings import *

MODEL_FILE = './imdb-model-cnn-1'
LOG_FILE = './log-model-cnn-1'

# Read back data
train_reviews = np.load("IMDB_train_fulltext_glove_X.npy")
train_labels = np.load("IMDB_train_fulltext_glove_y.npy")
test_reviews = np.load("IMDB_test_fulltext_glove_X.npy")
test_labels = np.load("IMDB_test_fulltext_glove_y.npy")

WV_FILE_GLOBAL = './data/wv/glove.42B.300d.120000-glovebox.pkl'

gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))

wv_size = gb_global.W.shape[1]

model = Sequential()
emb = Embedding(gb_global.W.shape[0], wv_size, weights=[gb_global.W],
                    input_length=train_reviews.shape[1])
emb.trainable = False
model.add(emb)
model.add(Convolution1D(64, 3, init='uniform'))
model.add(Activation('relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))

model.add(Convolution1D(64, 3, init='uniform'))
model.add(Activation('relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")

history = train_neural.train_sequential(model, train_reviews, train_labels, MODEL_FILE)
acc = train_neural.test_sequential(model, test_reviews, test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)
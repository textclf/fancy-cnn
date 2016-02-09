import cPickle as pickle

import numpy as np
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers.core import Dense, Activation

from nn import train_neural
from cnn.layers.embeddings import *

MODEL_FILE = './imdb-model-gru-1'
LOG_FILE = './log-model-gru'

# Read back data
train_reviews = np.load("IMDB_train_fulltext_glove_X.npy")
train_labels = np.load("IMDB_train_fulltext_glove_y.npy")
test_reviews = np.load("IMDB_test_fulltext_glove_X.npy")
test_labels = np.load("IMDB_test_fulltext_glove_y.npy")

WV_FILE_GLOBAL = './data/wv/glove.42B.300d.120000-glovebox.pkl'

gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))

wv_size = gb_global.W.shape[1]

model = Sequential()
model.add(make_embedding(vocab_size=gb_global.W.shape[0], init=gb_global.W, wv_size=wv_size,
                         fixed=True, constraint=None))
model.add(GRU(128, init='uniform'))
#model.add(Dropout(0.2))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

history = train_neural.train_model(model, train_reviews, train_labels, MODEL_FILE)
acc = train_neural.test_model(model, test_reviews, test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)
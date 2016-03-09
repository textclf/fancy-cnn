import cPickle as pickle
import numpy as np
import os
from os.path import join as path_join
import sys

from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

ROOT_PATH = '../..'
sys.path.append(ROOT_PATH)

from textclf.nn import train_neural
from textclf.nn.embeddings import make_embedding

MODEL_FILE = './models/3votes/useful/yelp-model-multi-rnn-1'
LOG_FILE = './outputs/3votes/useful/log-model-multi-rnn-1'

# Read back data
# train_reviews = np.load("../../Yelp_funny_train_fulltext_glove_300_X.npy")
# train_labels = np.load("../../Yelp_funny_train_fulltext_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_funny_test_fulltext_glove_300_X.npy")
# test_labels = np.load("../../Yelp_funny_test_fulltext_glove_300_y.npy")

train_reviews = np.load("../../Yelp_useful_train_fulltext_glove_300_X.npy")
train_labels = np.load("../../Yelp_useful_train_fulltext_glove_300_y.npy")
test_reviews = np.load("../../Yelp_useful_test_fulltext_glove_300_X.npy")
test_labels = np.load("../../Yelp_useful_test_fulltext_glove_300_y.npy")

# train_reviews = np.load("../../Yelp_cool_train_fulltext_glove_300_X.npy")
# train_labels = np.load("../../Yelp_cool_train_fulltext_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_cool_test_fulltext_glove_300_X.npy")
# test_labels = np.load("../../Yelp_cool_test_fulltext_glove_300_y.npy")

WV_FILE_GLOBAL = path_join(ROOT_PATH, 'embeddings/wv/glove.42B.300d.120000-glovebox.pkl')

gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))

wv_size = gb_global.W.shape[1]

model = Sequential()
model.add(make_embedding(vocab_size=gb_global.W.shape[0], init=gb_global.W, wv_size=wv_size,
                         fixed=True, constraint=None))
model.add(GRU(128, init='uniform', return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(64, init='uniform', return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(16, init='uniform'))
model.add(Dropout(0.2))
model.add(Dense(1, init='uniform'))
model.add(Activation('tanh'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

history = train_neural.train_sequential(model, train_reviews, train_labels, MODEL_FILE)
acc = train_neural.test_sequential(model, test_reviews, test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)

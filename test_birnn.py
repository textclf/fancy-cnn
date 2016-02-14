import cPickle as pickle

import numpy as np
from keras.layers.recurrent import GRU
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from nn import train_neural
from cnn.layers.embeddings import *

MODEL_FILE = './imdb-model-gru-1'
LOG_FILE = './log-model-gru'

# Read back data
train_reviews = np.load("IMDB_train_fulltext_glove_X.npy")
train_labels = np.load("IMDB_train_fulltext_glove_y.npy")


train_reviews = train_reviews[:50]
train_labels = train_labels[:50]

test_reviews = np.load("IMDB_test_fulltext_glove_X.npy")
test_labels = np.load("IMDB_test_fulltext_glove_y.npy")

test_reviews = test_reviews[:50]
test_labels = test_labels[:50]

WV_FILE_GLOBAL = './data/wv/glove.42B.300d.120000-glovebox.pkl'

gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))

wv_size = gb_global.W.shape[1]

model = Graph()
model.add_input('input', (len(train_reviews[0]), ), dtype='int')
model.add_node(make_embedding(vocab_size=gb_global.W.shape[0], init=gb_global.W, wv_size=wv_size,
                         fixed=True, constraint=None), name='wvs', input='input')
model.add_node(GRU(128, init='uniform'), name='gru_forwards', input='wvs')
model.add_node(GRU(128, go_backwards=True, init='uniform'), name='gru_backwards', input='wvs')
model.add_node(Dropout(0.5), name='gru_dropout', inputs=['gru_forwards', 'gru_backwards'])
model.add_node(Dense(1, init='uniform', activation='sigmoid'), name='probability', input='gru_dropout')
model.add_output(name='prediction', input='probability')

model.compile(loss={'prediction': 'binary_crossentropy'}, optimizer='adam')

fit_params = {
    "data": {
        'input': train_reviews,
        'prediction': train_labels
    },
    "batch_size": 32,
    "nb_epoch": 5,
    "verbose": True,
    "validation_split": 0.1,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_loss'),
                  ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=True, save_best_only=True)]
}

history = train_neural.train_graph(model, fit_params)
acc = train_neural.test_graph(model, {'input': test_reviews}, 'prediction', test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)
import cPickle as pickle
import logging
import numpy as np
from os.path import join as path_join
import sys

import keras.backend as K

from keras.models import Sequential, Graph
from keras.layers.containers import Graph as SubGraph
from keras.layers.containers import Sequential as Stack
from keras.layers.core import *
from keras.layers import Embedding
from keras.layers.convolutional import *
from keras.layers.recurrent import GRU, LSTM
from keras.regularizers import l2

ROOT_PATH = '../..'
sys.path.append(ROOT_PATH)

from textclf.nn import train_neural
from textclf.nn.timedistributed import TimeDistributed

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)


MODEL_FILE = './imdb-model-rcnn-1'
LOG_FILE = './log-model-rcnn-1'


# Read back data
WV_FILE_IMDB = path_join(ROOT_PATH, 'embeddings/wv/IMDB-GloVe-300dim-glovebox.pkl')
WV_FILE_GLOBAL = path_join(ROOT_PATH, 'embeddings/wv/glove.42B.300d.120000-glovebox.pkl')

gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))
gb_imdb = pickle.load(open(WV_FILE_IMDB, 'rb'))


train, test = {}, {}

log('Loading training data')

train['text4imdb'] = np.load(path_join(ROOT_PATH, 'IMDB_train_glove_X.npy'))
train['text4global'] = np.load(path_join(ROOT_PATH, 'IMDB_train_global_glove_X.npy'))
train['labels'] = np.load(path_join(ROOT_PATH, 'IMDB_train_glove_y.npy'))

log('Shuffling training data')
shuff = range(train['text4imdb'].shape[0])
np.random.shuffle(shuff)

for k in train.keys():
    train[k] = train[k][shuff]
    # -- flatten across paragraph dimension, will later be reconstructed in the embedding
    if 'lab' not in k:
        train[k] = train[k].reshape(train[k].shape[0], -1)

del shuff

log('Loading testing data')

# -- testing data
test['text4imdb'] = np.load(path_join(ROOT_PATH, 'IMDB_test_glove_X.npy'))
test['text4global'] = np.load(path_join(ROOT_PATH, 'IMDB_test_global_glove_X.npy'))
test['labels'] = np.load(path_join(ROOT_PATH, 'IMDB_test_glove_y.npy'))

test['text4imdb'] = test['text4imdb'].reshape(test['text4imdb'].shape[0], -1)
test['text4global'] = test['text4global'].reshape(test['text4global'].shape[0], -1)




log('Building model architecture...')
NGRAMS = [2, 3, 4, 5, 7]
NFILTERS = 32 * 3
SENTENCE_LENGTH = 50
PARAGRAPH_LENGTH = 50

INPUT_SHAPE = (SENTENCE_LENGTH * PARAGRAPH_LENGTH, )

global_vectors = Sequential([Embedding(input_dim=gb_global.W.shape[0], output_dim=300, weights=[gb_global.W], input_length=INPUT_SHAPE[0])])
imdb_vectors = Sequential([Embedding(input_dim=gb_imdb.W.shape[0], output_dim=300, weights=[gb_imdb.W], input_length=INPUT_SHAPE[0])])

model = Sequential()

model.add(Merge([global_vectors, imdb_vectors], mode='concat'))

model.add(Reshape((PARAGRAPH_LENGTH, SENTENCE_LENGTH, 2, 300)))
model.add(Permute(dims=(1, 3, 2, 4)))

# -- create convolution units...
conv_unit = SubGraph()
conv_unit = Graph()
conv_unit.add_input('embeddings', input_shape=model.output_shape[1:])

for n in NGRAMS:
    conv_unit.add_node(
        TimeDistributed(Convolution2D(NFILTERS, n, 300, 
            W_regularizer=l2(0.0001), 
            activation='relu')
        ), 
        name='conv{}gram'.format(n), input='embeddings'
    )

    conv_unit.add_node(
        TimeDistributed(MaxPooling2D(pool_size=(SENTENCE_LENGTH - n + 1, 1))),
        name='maxpool{}gram'.format(n), input='conv{}gram'.format(n)
    )
    
    # conv_unit.add_node(
    #         Lambda(
    #             function=lambda x: K.squeeze(x, axis=-1),
    #             output_shape=lambda s: s[:-1]
    #             ),
    #         name='squeeze{}gram'.format(n), input='conv{}gram'.format(n)
    # )

    # conv_unit.add_node(
    #     TimeDistributed(GRU(10), input_shape=conv_unit.nodes['squeeze{}gram'.format(n)].output_shape[1:]),
    #     name='gru-attn-forward{}gram'.format(n), input='squeeze{}gram'.format(n)
    # )
    
    # conv_unit.add_node(
    #     TimeDistributed(GRU(10, go_backwards=True)),
    #     name='gru-attn-backward{}gram'.format(n), input='squeeze{}gram'.format(n)
    # )

    conv_unit.add_node(
        Dropout(0.15),
        name='dropout{}gram'.format(n), input='maxpool{}gram'.format(n)
    )

    conv_unit.add_node(
        TimeDistributed(Flatten()), 
        name='flatten{}gram'.format(n), input='dropout{}gram'.format(n)
    )


    # conv_unit.add_node(
    #     Dropout(0.15),
    #     name='dropout-gru{}gram'.format(n), input='gru-attn-forward{}gram'.format(n)
    # )


    conv_unit.add_node(
        TimeDistributed(Highway(activation='relu')), 
        name='highway{}gram'.format(n), 
        input='flatten{}gram'.format(n)
        # inputs=['flatten{}gram'.format(n), 'dropout-gru{}gram'.format(n)]
    )

# -- merge across all the n-gram sizes
conv_unit.add_node(Dropout(0.1), name='dropout', inputs=['highway{}gram'.format(n) for n in NGRAMS])

# -- add a bidirectional RNN
conv_unit.add_node(GRU(100), name='forwards', input='dropout', concat_axis=-1)
conv_unit.add_node(GRU(100, go_backwards=True), name='backwards', input='dropout', concat_axis=-1)

conv_unit.add_node(Dropout(0.7), name='gru_dropout', inputs=['forwards', 'backwards'], create_output=True)

model.add(conv_unit)

model.add(MaxoutDense(128, 8))

model.add(Dropout(0.5))

model.add(Highway(activation='relu'))
model.add(Highway(activation='relu'))
model.add(Highway(activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

log('Compiling model (may take >10 mins)')

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')



log('Testing model!')
# -- since we are using a merge model, we need to make a list
train_reviews = [train['text4global'], train['text4imdb']]
test_reviews = [test['text4global'], test['text4imdb']]

train_labels = train['labels']
test_labels = test['labels']


history = train_neural.train_sequential(model, train_reviews, train_labels, MODEL_FILE)
acc = train_neural.test_sequential(model, test_reviews, test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)



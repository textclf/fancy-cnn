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
from keras.callbacks import EarlyStopping, ModelCheckpoint

ROOT_PATH = '../..'
sys.path.append(ROOT_PATH)

from textclf.nn import train_neural
from textclf.nn.timedistributed import TimeDistributed
from textclf.wordvectors.char import CharMapper

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

TARGET = 'cool'

MODEL_FILE = './yelp-model-rcnn-char-big-{}'.format(TARGET)
LOG_FILE = './log-model-rcnn-char-big-{}'.format(TARGET)

log('Loading training data')

train_reviews = np.load(path_join(ROOT_PATH, 'Yelp_{}_sentences_train_char_3votes_X.npy'.format(TARGET)))
train_labels = np.load(path_join(ROOT_PATH, 'Yelp_{}_sentences_train_char_3votes_y.npy'.format(TARGET)))

log('Shuffling training data')
nb_samples = train_reviews.shape[0]
shuff = range(nb_samples)
np.random.shuffle(shuff)

train_reviews, train_labels = train_reviews[shuff].reshape(nb_samples, -1), train_labels[shuff]
del shuff

log('Loading testing data')

# -- testing data

test_reviews = np.load(path_join(ROOT_PATH, 'Yelp_{}_sentences_test_char_3votes_X.npy'.format(TARGET)))
test_reviews = test_reviews.reshape(test_reviews.shape[0], -1)

test_labels = np.load(path_join(ROOT_PATH, 'Yelp_{}_sentences_test_char_3votes_y.npy'.format(TARGET)))




log('Building model architecture...')
NGRAMS = [1, 2, 3, 4, 5]
NFILTERS = 32 * 3

CHARACTERS_PER_WORD = 15
WORDS_PER_DOCUMENT = 300
NUMBER_CHARACTERS = len(CharMapper.ALLOWED_CHARS) + 2
EMBEDDING_DIM = 100
INPUT_SHAPE = (CHARACTERS_PER_WORD * WORDS_PER_DOCUMENT, )


model = Sequential()

model.add(Embedding(input_dim=NUMBER_CHARACTERS, output_dim=EMBEDDING_DIM, input_length=INPUT_SHAPE[0]))

model.add(Reshape((WORDS_PER_DOCUMENT, CHARACTERS_PER_WORD, EMBEDDING_DIM)))

# -- create convolution units...
conv_unit = Graph()
conv_unit.add_input('embeddings', input_shape=model.output_shape[1:])

for n in NGRAMS:

    # -- convolve over the character n-gram in each word
    conv_unit.add_node(
        TimeDistributed(Convolution1D(NFILTERS, n, 
    #        W_regularizer=l2(0.0001), 
            activation='relu')
        ), 
        name='conv{}gram'.format(n), input='embeddings'
    )

    # -- take a maxpool over the output convolved "plane"
    conv_unit.add_node(
        TimeDistributed(MaxPooling1D(CHARACTERS_PER_WORD - n + 1)),
        name='maxpool{}gram'.format(n), input='conv{}gram'.format(n)
    )
    
    # -- we need to squeeze out a tensordim of val 1
    conv_unit.add_node(
        TimeDistributed(Flatten()), 
        name='flattenmaxpool{}gram'.format(n), input='maxpool{}gram'.format(n)
    )

    # -- have a bidirectional LSTM over the convolved character plane
    #conv_unit.add_node(
    #    TimeDistributed(LSTM(10)),
    #    name='forwardlstm{}gram'.format(n), input='conv{}gram'.format(n)
    #)
    #conv_unit.add_node(
    #    TimeDistributed(LSTM(10, go_backwards=True)),
    #    name='backwardlstm{}gram'.format(n), input='conv{}gram'.format(n)
    #)

    # -- concat the maxpool, fwd-lstm, and bwd-lstm --> dropout
    #conv_unit.add_node(
    #    Dropout(0.15),
    #    name='dropout{}gram'.format(n), inputs=['flattenmaxpool{}gram'.format(n), 'forwardlstm{}gram'.format(n), 'backwardlstm{}gram'.format(n)]
    #)

    # -- use a highway layer as the final per-ngram feature
    #conv_unit.add_node(
    #    TimeDistributed(Highway(activation='relu')), 
    #    name='highway{}gram'.format(n), 
    #    input='dropout{}gram'.format(n)
    #)

# -- merge across all the n-gram sizes
conv_unit.add_node(Dropout(0.5), name='dropout', inputs=['flattenmaxpool{}gram'.format(n) for n in NGRAMS])

# -- add a bidirectional RNN
conv_unit.add_node(GRU(90), name='forwards', input='dropout', concat_axis=-1)
conv_unit.add_node(GRU(90, go_backwards=True), name='backwards', input='dropout', concat_axis=-1)

conv_unit.add_node(Dropout(0.5), name='gru_dropout', inputs=['forwards', 'backwards'], create_output=True)

model.add(conv_unit)

# model.add(MaxoutDense(128, 8))

# model.add(Dropout(0.5))

# model.add(Highway(activation='relu'))
# model.add(Highway(activation='relu'))
model.add(Highway(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

log('Compiling model (may take >10 mins)')

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')



log('Training/Testing model!')

fit_params = {
    "batch_size": 64,
    "nb_epoch": 100,
    "verbose": True,
    "validation_split": 0.4,
    "show_accuracy": True,
    "callbacks": [EarlyStopping(verbose=True, patience=12, monitor='val_acc'),
                  ModelCheckpoint(MODEL_FILE, monitor='val_acc', verbose=True, save_best_only=True)]
}

history = train_neural.train_sequential(model, train_reviews, train_labels, MODEL_FILE, fit_params=fit_params)
acc = train_neural.test_sequential(model, test_reviews, test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)



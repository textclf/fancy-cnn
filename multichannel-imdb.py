import logging
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, model_from_json, Graph
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD

from cnn.layers.convolutions import *
from cnn.layers.embeddings import *

import numpy as np
import cPickle as pickle

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)


if __name__ == '__main__':

    

    WV_FILE = './data/wv/IMDB-GloVe-300dim-glovebox.pkl'
    WV_FILE_GLOBAL = './data/wv/glove.42B.300d.120000-glovebox.pkl'

    MODEL_FILE = './test-model.h5'

    # -- load in all the data
    train, test = {}, {}

    log('Loading training data')

    train['text4imdb'] = np.load('IMDB_train_glove_X.npy')
    train['text4global'] = np.load('IMDB_train_global_glove_X.npy')
    train['labels'] = np.load('IMDB_train_glove_y.npy')

    log('Shuffling training data')
    shuff = range(train['text4imdb'].shape[0])
    np.random.shuffle(shuff)

    for k in train.keys():
        train[k] = train[k][shuff]
        if 'lab' not in k:
            train[k] = train[k].reshape(train[k].shape[0], -1)

    # -- flatten across paragraph dimension, will later be reconstructed in the embedding
    

    weights = 1.0 * ((train['text4imdb'] > 0) | (train['text4global'] > 0))

    del shuff

    log('Loading testing data')

    # -- testing data
    test['text4imdb'] = np.load('IMDB_test_glove_X.npy')
    test['text4imdb'] = test['text4imdb'].reshape(test['text4imdb'].shape[0], -1)

    test['text4global'] = np.load('IMDB_test_global_glove_X.npy')
    test['text4global'] = test['text4global'].reshape(test['text4global'].shape[0], -1)

    test['labels'] = np.load('IMDB_test_glove_y.npy')

    log('Loading IMDB trained word vectors')
    gb = pickle.load(open(WV_FILE, 'rb'))

    log('Loading pretrained word vectors')
    gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))
    
    WV_PARAMS = {
        'imdb_vectors' :
        {
            'input_name' : 'imdb_input',
            'vocab_size' : gb.W.shape[0],
            'init' : gb.W,
            'fixed' : False
        },
        'glove_vectors' :
        {
            'input_name' : 'glove_input',
            'vocab_size' : gb_global.W.shape[0],
            'init' : gb_global.W,
            'fixed' : False
        },
        'fixed_glove_vectors' : 
        {
            'input_name' : 'glove_input',
            'vocab_size' : gb_global.W.shape[0],
            'init' : gb_global.W,
            'fixed' : True
        }
    }

    NGRAMS = [1, 3, 4, 5, 6, 7, 9]
    NFILTERS = 64
    SENTENCE_LENGTH = 50
    PARAGRAPH_LENGTH = 50

    log('Making graph model')
    graph = Graph()

    log('Making embedding')
    
    seen_inputs = set()

    for name, params in WV_PARAMS.iteritems():
        # -- add each word vector channel
        if params['input_name'] not in seen_inputs:
            seen_inputs.add(params['input_name'])
            graph.add_input(params['input_name'], (SENTENCE_LENGTH * PARAGRAPH_LENGTH, ), dtype='int')

        # -- create the embedding!
        graph.add_node(make_embedding(wv_size=300, **params), name=name, input=params['input_name'])

    # -- reshape to 5D tensor
    graph.add_node(Reshape((PARAGRAPH_LENGTH, SENTENCE_LENGTH, len(WV_PARAMS), 300)), name='reshape', inputs=WV_PARAMS.keys(), merge_mode='concat')
   
    # -- permut
    graph.add_node(Permute(dims=(1, 3, 2, 4)), name='embedding', input='reshape')

    log('Adding convolved n-grams')
    # for n in [4, 5]:
    for n in NGRAMS:
        graph.add_node(
            TimeDistributedConvolution2D(NFILTERS, n, WV_PARAMS['glove_vectors']['init'].shape[1], activation='relu'), 
            name='conv{}gram'.format(n), input='embedding')

        graph.add_node(
            TimeDistributedMaxPooling2D(pool_size=(SENTENCE_LENGTH - n + 1, 1)),
            name='maxpool{}gram'.format(n), input='conv{}gram'.format(n))

        graph.add_node(
            Dropout(0.79),
            name='dropout{}gram'.format(n), input='maxpool{}gram'.format(n))    

        graph.add_node(
            TimeDistributedFlatten(), 
            name='flatten{}gram'.format(n), input='dropout{}gram'.format(n))

    log('Adding bi-directional GRU')
    graph.add_node(GRU(45), name='gru_forwards', inputs=['flatten{}gram'.format(n) for n in NGRAMS], concat_axis=-1)
    graph.add_node(GRU(45, go_backwards=True), name='gru_backwards', inputs=['flatten{}gram'.format(n) for n in NGRAMS], concat_axis=-1)
    # graph.add_node(GRU(16), name='gru', input='flatten4gram')

    ADDITIONAL_FC = False

    graph.add_node(Dropout(0.7), name='gru_dropout', inputs=['gru_forwards', 'gru_backwards'])

    if ADDITIONAL_FC:

        graph.add_node(MaxoutDense(32, 8, init='he_normal'), name='maxout', input='gru_dropout')

        graph.add_node(Dropout(0.5), name='maxout_dropout', input='maxout')

        graph.add_node(Dense(1, activation='sigmoid'), name='probability', input='maxout_dropout')
    else:
        graph.add_node(Dense(1, activation='sigmoid'), name='probability', input='gru_dropout')

    graph.add_output(name='prediction', input='probability')

    log('Compiling model (Veuillez patienter)...')
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0001, nesterov=True)
    # graph.compile(sgd, {'prediction': 'binary_crossentropy'})
    graph.compile('rmsprop', {'prediction': 'binary_crossentropy'})

    log('Fitting! Hit CTRL-C to stop early...')
    try:
        history = graph.fit(
            {
                'imdb_input' : train['text4imdb'], 
                'glove_input' : train['text4global'], 
                'prediction': train['labels']
            }, 
            validation_split=0.35, batch_size=16, nb_epoch=100, 
            verbose=True, # -- for logging purposes
            sample_weight = {'prediction' : weights}, 
            callbacks = 
                   [
                       EarlyStopping(verbose=True, patience=30, monitor='val_loss'),
                       ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=True, save_best_only=True)
                   ]
           )
    except KeyboardInterrupt:
        log('Training stopped early!')

    log('Loading best weights...')
    graph.load_weights(MODEL_FILE)

    log('getting predictions on the test set')
    yhat = graph.predict({'imdb_input' : test['text4imdb'], 'glove_input' : test['text4global'], }, verbose=True, batch_size=50)

    acc = ((yhat['prediction'].ravel() > 0.5) == (test['labels'] > 0.5)).mean()

    log('Test set accuracy of {}%.'.format(acc * 100.0))
    log('Test set error of {}%. Exiting...'.format((1 - acc) * 100.0))
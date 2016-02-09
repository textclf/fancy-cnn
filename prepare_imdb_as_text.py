"""
prepare-imdb.py

description: prepare the imdb data for training in DNNs
"""
import cPickle as pickle
import logging

import numpy as np

from nlpdatahandlers import ImdbDataHandler
from wordvectors.glove import GloVeBox

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

IMDB_DATA = './datasets/aclImdb/aclImdb'
IMDB_WV_FILE = './data/wv/IMDB-GloVe-300dim.txt'
GLOBAL_WV_FILE = './data/wv/glove.42B.300d.120000.txt'
WORDS_PER_TEXT = 300

if __name__ == '__main__':

    log('Building global word vectors from {}'.format(GLOBAL_WV_FILE))
    global_gb = GloVeBox(GLOBAL_WV_FILE)
    global_gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    log('writing GloVeBox pickle...')
    pickle.dump(global_gb, open(GLOBAL_WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    log('Load data from original source')
    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)

    log('Converting to global word vectors - train')
    reviews_wvs_train = imdb.to_word_level_idx(train_reviews, global_gb, WORDS_PER_TEXT)
    # -- training data save
    np.save('IMDB_train_fulltext_glove_X.npy', reviews_wvs_train)
    np.save('IMDB_train_fulltext_glove_y.npy', train_labels)

    del reviews_wvs_train

    log('Converting to global word vectors - test')
    reviews_wvs_test = imdb.to_word_level_idx(test_reviews, global_gb, WORDS_PER_TEXT)
    # -- testing data save
    np.save('IMDB_test_fulltext_glove_X.npy', reviews_wvs_test)
    np.save('IMDB_test_fulltext_glove_y.npy', test_labels)

    del reviews_wvs_test
"""
prepare-imdb.py

description: prepare the imdb data for training in DNNs
"""
from nlpdatahandlers import ImdbDataHandler

import cPickle as pickle
import logging

import numpy as np

from wordvectors.char import CharMapper

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

IMDB_DATA = './datasets/aclImdb/aclImdb'

CHARACTERS_PER_WORD = 15
WORDS_PER_DOCUMENT = 300
PREPEND = False

if __name__ == '__main__':

    log('Initializing CharMapper')
    cm = CharMapper()

    log('Load data from original source')
    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)

    log('Converting to character level representations')
    train_global_wvs_reviews = imdb.to_char_level_idx(train_reviews, 
        char_container=cm,
        chars_per_word=CHARACTERS_PER_WORD,
        words_per_document=WORDS_PER_DOCUMENT,
        prepend=PREPEND)

    test_global_wvs_reviews = imdb.to_char_level_idx(test_reviews, 
        char_container=cm,
        chars_per_word=CHARACTERS_PER_WORD,
        words_per_document=WORDS_PER_DOCUMENT,
        prepend=PREPEND)

    log('saving data')
    # -- training data save
    np.save('IMDB_train_char_X.npy', train_global_wvs_reviews)
    np.save('IMDB_train_char_y.npy', train_labels)

    # -- testing data save
    np.save('IMDB_test_char_X.npy', test_global_wvs_reviews)
    np.save('IMDB_test_char_y.npy', test_labels)
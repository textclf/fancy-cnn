"""
prepare_yelp_char.py

description: prepare the yelp data for training in convolutional recurrent architectures over characters
"""
from nlpdatahandlers import YelpDataHandler

import cPickle as pickle
import logging

import numpy as np

from textclf.wordvectors.char import CharMapper

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

YELP_USEFUL_TRAIN = '../yelp-dataset/TrainSet_useful_185292'
YELP_USEFUL_DEV = '../yelp-dataset/DevSet_useful_185292'
YELP_USEFUL_TEST = '../yelp-dataset/TestSet_useful_185292'

YELP_FUNNY_TRAIN = '../yelp-dataset/TrainSet_funny_75064'
YELP_FUNNY_DEV = '../yelp-dataset/DevSet_funny_75064'
YELP_FUNNY_TEST = '../yelp-dataset/TestSet_funny_75064'

YELP_COOL_TRAIN = '../yelp-dataset/TrainSet_cool_88698'
YELP_COOL_DEV = '../yelp-dataset/DevSet_cool_88698'
YELP_COOL_TEST = '../yelp-dataset/TestSet_cool_88698'

CHARACTERS_PER_WORD = 15
WORDS_PER_DOCUMENT = 300
PREPEND = False

if __name__ == '__main__':

    log('Initializing CharMapper')
    cm = CharMapper()

    yelp = YelpDataHandler()

    def get_yelp_char(train_reviews, test_reviews):
        log('Converting to character level representations')
        train_reviews = yelp.to_char_level_idx(train_reviews, 
            char_container=cm,
            chars_per_word=CHARACTERS_PER_WORD,
            words_per_document=WORDS_PER_DOCUMENT,
            prepend=PREPEND)

        test_reviews = yelp.to_char_level_idx(test_reviews, 
            char_container=cm,
            chars_per_word=CHARACTERS_PER_WORD,
            words_per_document=WORDS_PER_DOCUMENT,
            prepend=PREPEND)
        return train_reviews, test_reviews

    ##################################
    ### YELP USEFUL
    ##################################
    log('Creating "useful" reviews sentence-datasets')
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_USEFUL_TRAIN, YELP_USEFUL_DEV, YELP_USEFUL_TEST)

    train_reviews, test_reviews = get_yelp_char(train_reviews, test_reviews)


    # -- training data save
    
    np.save('Yelp_useful_sentences_train_char_X.npy', train_reviews)
    np.save('Yelp_useful_sentences_train_char_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_useful_sentences_test_char_X.npy', test_reviews)
    np.save('Yelp_useful_sentences_test_char_y.npy', test_labels)

    ##################################
    ### YELP FUNNY
    ##################################
    log('Creating "funny" reviews sentence-datasets')
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_FUNNY_TRAIN, YELP_FUNNY_DEV, YELP_FUNNY_TEST)

    train_reviews, test_reviews = get_yelp_char(train_reviews, test_reviews)

    # -- training data save
    
    np.save('Yelp_funny_sentences_train_char_X.npy', train_reviews)
    np.save('Yelp_funny_sentences_train_char_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_funny_sentences_test_char_X.npy', test_reviews)
    np.save('Yelp_funny_sentences_test_char_y.npy', test_labels)

    ##################################
    ### YELP COOL
    ##################################
    log('Creating "cool" reviews sentence-datasets')
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_COOL_TRAIN, YELP_COOL_DEV, YELP_COOL_TEST)

    train_reviews, test_reviews = get_yelp_char(train_reviews, test_reviews)

    # -- training data save
    
    np.save('Yelp_cool_sentences_train_char_X.npy', train_reviews)
    np.save('Yelp_cool_sentences_train_char_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_cool_sentences_test_char_X.npy', test_reviews)
    np.save('Yelp_cool_sentences_test_char_y.npy', test_labels)

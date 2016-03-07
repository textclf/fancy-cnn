"""
prepare_yelp_sentences.py

description: prepare the yelp data for training in convolutional recurrent architectures over sentences
"""
from nlpdatahandlers import YelpDataHandler

import cPickle as pickle
import logging

import numpy as np

from textclf.wordvectors.glove import GloVeBox

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

GLOBAL_WV_FILE = './embeddings/wv/glove.42B.300d.120000.txt'
YELP_WV_FILE = './embeddings/wv/Yelp-GloVe-300dim.txt'
WORDS_PER_SENTENCE = 50
SENTENCES_PER_PARAGRAPH = 50
PREPEND = False

if __name__ == '__main__':

    log('Building word vectors from {}'.format(YELP_WV_FILE))
    yelp_gb = GloVeBox(YELP_WV_FILE)
    yelp_gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    log('Building global word vectors from {}'.format(GLOBAL_WV_FILE))
    global_gb = GloVeBox(GLOBAL_WV_FILE)
    global_gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    log('writing GloVeBox pickle...')
    pickle.dump(yelp_gb, open(YELP_WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(global_gb, open(GLOBAL_WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    yelp = YelpDataHandler()

    ##################################
    ### YELP USEFUL
    ##################################
    log('Creating "useful" reviews sentence-datasets')
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_USEFUL_TRAIN, YELP_USEFUL_DEV, YELP_USEFUL_TEST)

    log('Converting to sentences: global word vectors')
    train_global_wvs_reviews = yelp.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, global_gb)
    test_global_wvs_reviews = yelp.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, global_gb)

    log('Converting to sentences: yelp word vectors')
    train_yelp_wvs_reviews = yelp.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, yelp_gb)
    test_yelp_wvs_reviews = yelp.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, yelp_gb)

    # -- training data save
    np.save('Yelp_useful_sentences_train_yelp_glove_X.npy', train_yelp_wvs_reviews)
    np.save('Yelp_useful_sentences_train_global_glove_X.npy', train_global_wvs_reviews)
    np.save('Yelp_useful_sentences_train_glove_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_useful_sentences_test_yelp_glove_X.npy', test_yelp_wvs_reviews)
    np.save('Yelp_useful_sentences_test_global_glove_X.npy', test_global_wvs_reviews)
    np.save('Yelp_useful_sentences_test_glove_y.npy', test_labels)

    ##################################
    ### YELP FUNNY
    ##################################
    log('Creating "funny" reviews sentence-datasets')
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_FUNNY_TRAIN, YELP_FUNNY_DEV, YELP_FUNNY_TEST)

    log('Converting to sentences: global word vectors')
    train_global_wvs_reviews = yelp.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, global_gb)
    test_global_wvs_reviews = yelp.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, global_gb)

    log('Converting to sentences: yelp word vectors')
    train_yelp_wvs_reviews = yelp.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, yelp_gb)
    test_yelp_wvs_reviews = yelp.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, yelp_gb)

    # -- training data save
    np.save('Yelp_funny_sentences_train_yelp_glove_X.npy', train_yelp_wvs_reviews)
    np.save('Yelp_funny_sentences_train_global_glove_X.npy', train_global_wvs_reviews)
    np.save('Yelp_funny_sentences_train_glove_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_funny_sentences_test_yelp_glove_X.npy', test_yelp_wvs_reviews)
    np.save('Yelp_funny_sentences_test_global_glove_X.npy', test_global_wvs_reviews)
    np.save('Yelp_funny_sentences_test_glove_y.npy', test_labels)

    ##################################
    ### YELP COOL
    ##################################
    log('Creating "cool" reviews sentence-datasets')
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_COOL_TRAIN, YELP_COOL_DEV, YELP_COOL_TEST)

    log('Converting to sentences: global word vectors')
    train_global_wvs_reviews = yelp.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, global_gb)
    test_global_wvs_reviews = yelp.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, global_gb)

    log('Converting to sentences: yelp word vectors')
    train_yelp_wvs_reviews = yelp.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, yelp_gb)
    test_yelp_wvs_reviews = yelp.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, yelp_gb)

    # -- training data save
    np.save('Yelp_cool_sentences_train_yelp_glove_X.npy', train_yelp_wvs_reviews)
    np.save('Yelp_cool_sentences_train_global_glove_X.npy', train_global_wvs_reviews)
    np.save('Yelp_cool_sentences_train_glove_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_cool_sentences_test_yelp_glove_X.npy', test_yelp_wvs_reviews)
    np.save('Yelp_cool_sentences_test_global_glove_X.npy', test_global_wvs_reviews)
    np.save('Yelp_cool_sentences_test_glove_y.npy', test_labels)

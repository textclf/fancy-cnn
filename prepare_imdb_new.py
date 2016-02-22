"""
prepare-imdb.py

description: prepare the imdb data for training in DNNs
"""
from nlpdatahandlers import ImdbDataHandler

import cPickle as pickle
import logging

import numpy as np

from textclf.wordvectors.glove import GloVeBox

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

IMDB_DATA = './datasets/aclImdb/aclImdb'
IMDB_WV_FILE = './embeddings/wv/IMDB-GloVe-300dim.txt'
GLOBAL_WV_FILE = './embeddings/wv/glove.42B.300d.120000.txt'
WORDS_PER_SENTENCE = 50
SENTENCES_PER_PARAGRAPH = 50
PREPEND = False

if __name__ == '__main__':

    log('Building word vectors from {}'.format(IMDB_WV_FILE))
    gb = GloVeBox(IMDB_WV_FILE)
    gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    log('Building global word vectors from {}'.format(GLOBAL_WV_FILE))
    global_gb = GloVeBox(GLOBAL_WV_FILE)
    global_gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    log('writing GloVeBox pickle...')
    pickle.dump(gb, open(IMDB_WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(global_gb, open(GLOBAL_WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    log('Load data from original source')
    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)

    log('Converting to sentences: global word vectors')
    train_global_wvs_reviews = imdb.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, global_gb)
    test_global_wvs_reviews = imdb.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, global_gb)

    log('Converting to sentences: only imdb word vectors')
    train_imdb_wvs_reviews = imdb.to_sentence_level_idx(train_reviews, SENTENCES_PER_PARAGRAPH,
                                                    WORDS_PER_SENTENCE, gb)
    test_imdb_wvs_reviews = imdb.to_sentence_level_idx(test_reviews, SENTENCES_PER_PARAGRAPH,
                                                   WORDS_PER_SENTENCE, gb)

    # -- training data save
    np.save('IMDB_train_glove_X.npy', train_imdb_wvs_reviews)
    np.save('IMDB_train_global_glove_X.npy', train_global_wvs_reviews)
    np.save('IMDB_train_glove_y.npy', train_labels)

    # -- testing data save
    np.save('IMDB_test_glove_X.npy', test_imdb_wvs_reviews)
    np.save('IMDB_test_global_glove_X.npy', test_global_wvs_reviews)
    np.save('IMDB_test_glove_y.npy', test_labels)
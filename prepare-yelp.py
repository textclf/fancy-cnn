'''
prepare-yelp.py

description: prepare the imdb data for training in DNNs
'''

import cPickle as pickle
import os
import glob
import logging
import re
from multiprocessing import Pool

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

import numpy as np

log('Importing spaCy...')
from spacy.en import English

from cnn.wordvectors.glove import GloVeBox
from util.misc import normalize_sos


log('Initializing spaCy...')
nlp = English()

# -- path where the download script downloads to
TRAIN_FILE = "datasets/yelp/data_funny_binary_balanced/TrainSet_147444"
DEV_FILE = "datasets/yelp/data_funny_binary_balanced/DevSet_147444"
TEST_FILE = "datasets/yelp/data_funny_binary_balanced/TestSet_147444"

NUM_TRAIN_REVIEWS = None # None if want to use all
NUM_TEST_REVIEWS = None

WV_FILE = './data/wv/Yelp-GloVe-300dim.txt'
#GLOBAL_WV_FILE = './data/wv/glove.42B.300d.120000.txt'


def parallel_run(f, parms):
    '''
    performs multi-core map of the function `f`
    over the parameter space spanned by parms.

    `f` MUST take only one argument. 
    '''
    pool = Pool()
    ret = pool.map(f, parms)
    pool.close()
    pool.join()
    return ret

def parse_paragraph(txt):
    '''
    Takes a text and returns a list of lists of tokens, where each sublist is a sentence
    '''
    return [[t.text for t in s] for s in nlp(re.sub('\s+', ' ', txt).strip()).sents]

def parse_tokens(txt):
    '''
    Takes a text and returns a list of tokens
    '''
    return [tx for tx in (t.text for t in nlp(u'' + txt.decode('ascii',errors='ignore'))) if tx != '\n']



if __name__ == '__main__':

    log('Loading Yelp Humor data')
    
    # -- construct wordvector instances
    log('Building word vectors from {}'.format(WV_FILE))
    gb = GloVeBox(WV_FILE)
    gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)#.index()

 #   log('Building global word vectors from {}'.format(GLOBAL_WV_FILE))
 #   global_gb = GloVeBox(GLOBAL_WV_FILE)
 #   global_gb.build(zero_token=True, normalize_variance=False, normalize_norm=True)#.index()

    log('writing GloVeBox pickle...')
    pickle.dump(gb, open(WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
#    pickle.dump(global_gb, open(GLOBAL_WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    log('Loading data...')
    with open(TRAIN_FILE) as file:
        [train_reviews, train_labels] = pickle.load(file)
    with open(DEV_FILE) as file:
        [dev_reviews, dev_labels] = pickle.load(file)

    # -- parameters to tune and set
    WORDS_PER_SENTENCE = 50
    SENTENCES_PER_PARAGRAPH = 50
    PREPEND = False

    log('Merging data...')
    # Merge train and dev
    train_reviews.extend(dev_reviews)
    train_reviews = train_reviews[:NUM_TRAIN_REVIEWS]
    train_labels.extend(dev_labels)
    train_labels = np.asarray(train_labels[:NUM_TRAIN_REVIEWS])


    log('Splitting training data into paragraphs')
    train_text_sentences = parallel_run(parse_paragraph, train_reviews)


    log('normalizing training inputs...')

    log('  --> building local word vector representation')
    train_repr = normalize_sos(
                            [
                                normalize_sos(review, WORDS_PER_SENTENCE, prepend=PREPEND) 
                                for review in gb.get_indices(train_text_sentences)
                            ], 
            SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE, PREPEND
        )

    train_text = np.array(train_repr)

#    log('  --> building global word vector representation')
#    global_train_repr = normalize_sos(
#                            [
#                                normalize_sos(review, WORDS_PER_SENTENCE, prepend=PREPEND) 
#                                for review in global_gb.get_indices(train_text_sentences)
#                            ], 
#            SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE, PREPEND
#        )

#    global_train_text = np.array(global_train_repr)

    # -- training data save
    np.save('Yelp_train_glove_X.npy', train_text)
 #   np.save('Yelp_train_global_glove_X.npy', global_train_text)
    np.save('Yelp_train_glove_y.npy', train_labels)

    with open(TEST_FILE) as file:
        [test_reviews, test_labels] = pickle.load(file)

    test_reviews = test_reviews[:NUM_TEST_REVIEWS]
    test_labels = np.asarray(test_labels[:NUM_TEST_REVIEWS])
    test_text_sentences = parallel_run(parse_paragraph, test_reviews)


    log('normalizing testing inputs...')
    log('  --> building local word vector representation')
    test_repr = normalize_sos(
                        [
                            normalize_sos(review, WORDS_PER_SENTENCE, prepend=PREPEND) 
                            for review in gb.get_indices(test_text_sentences)
                        ], 
        SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE, PREPEND
    )
    test_text = np.array(test_repr)
#    log('  --> building global word vector representation')
  #  global_test_repr = normalize_sos(
  #                      [
  #                          normalize_sos(review, WORDS_PER_SENTENCE, prepend=PREPEND) 
  #                          for review in global_gb.get_indices(test_text_sentences)
  #                      ], 
  #      SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE, PREPEND
  #  )

  #  global_test_text = np.array(global_test_repr)


    # -- testing data save
    np.save('Yelp_test_glove_X.npy', test_text)
  #  np.save('Yelp_test_global_glove_X.npy', global_test_text)
    np.save('Yelp_test_glove_y.npy', test_labels)
    

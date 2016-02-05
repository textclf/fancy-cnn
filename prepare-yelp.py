'''
prepare-yelp.py

description: prepare the yelp data for training in DNNs
'''

import cPickle as pickle
import logging
from multiprocessing import Pool

import numpy as np

from wordvectors.glove import GloVeBox
from util.misc import normalize_sos

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

def parse_paragraph(txt):
    '''
    Takes a text and returns a list of lists of tokens, where each sublist is a sentence
    '''
    return [[t.text for t in s] for s in nlp(txt).sents]

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

TRAIN_FILE = "datasets/yelp/data_funny_binary_balanced/TrainSet_147444"
DEV_FILE = "datasets/yelp/data_funny_binary_balanced/DevSet_147444"
TEST_FILE = "datasets/yelp/data_funny_binary_balanced/TestSet_147444"

NUM_TRAIN_REVIEWS = None # None if want to use all
NUM_TEST_REVIEWS = None

# -- parameters to tune and set
WORDS_PER_SENTENCE = 20
SENTENCES_PER_PARAGRAPH = 20

WV_FILE = './data/wv/glove.42B.300d.120000.txt'

log('Importing spaCy...')
from spacy.en import English

log('Initializing spaCy...')
nlp = English()

if __name__ == '__main__':

    log('Building word vectors from {}'.format(WV_FILE))
    gb = GloVeBox(WV_FILE)
    gb.build(zero_token=True).index()

    log('writing GloVeBox pickle...')
    pickle.dump(gb, open(WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    log('Loading train and test pickles...')

    with open(TRAIN_FILE) as file:
        [train_reviews, train_labels] = pickle.load(file)
    with open(DEV_FILE) as file:
        [dev_reviews, dev_labels] = pickle.load(file)
    with open(TEST_FILE) as file:
        [test_reviews, test_labels] = pickle.load(file)

    # Merge train and dev
    train_reviews.extend(dev_reviews)
    train_reviews = train_reviews[:NUM_TRAIN_REVIEWS]
    train_labels.extend(dev_labels)
    train_labels = train_labels[:NUM_TRAIN_REVIEWS]

    test_reviews = test_reviews[:NUM_TEST_REVIEWS]
    test_labels = test_labels[:NUM_TEST_REVIEWS]

    log('Splitting training data into paragraphs')
    train_text_sentences = parallel_run(parse_paragraph, train_reviews)
    test_text_sentences = parallel_run(parse_paragraph, test_reviews)

    log('normalizing training inputs...')
    train_repr = normalize_sos(
                        [
                                normalize_sos(review, WORDS_PER_SENTENCE)
                                for review in gb.get_indices(train_text_sentences)
                        ],
            SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE
        )

    train_text = np.array(train_repr)

    log('normalizing testing inputs...')
    test_repr = normalize_sos(
                        [
                            normalize_sos(review, WORDS_PER_SENTENCE)
                            for review in gb.get_indices(test_text_sentences)
                        ],
        SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE
    )

    test_text = np.array(test_repr)

    log('Saving...')

    # -- training data save
    np.save('Yelp_train_glove_X.npy', train_text)
    np.save('Yelp_train_glove_y.npy', train_labels)

    # -- testing data save
    np.save('Yelp_test_glove_X.npy', test_text)
    np.save('Yelp_test_glove_y.npy', test_labels)







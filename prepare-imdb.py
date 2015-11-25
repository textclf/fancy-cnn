'''
prepare-imdb.py

description: prepare the imdb data for training in DNNs
'''

import cPickle as pickle
import os
import glob
import logging
from multiprocessing import Pool

import numpy as np
from spacy.en import English

from cnn.wordvectors.glove import GloVeBox
from utilities.misc import normalize_sos

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg, logger=logger):
    logger.info(LOGGER_PREFIX % msg)

log('Initializing spaCy...')
nlp = English()

# -- path where the download script downloads to
DATA_PREFIX = './datasets/aclImdb/aclImdb'
WV_FILE = './data/wv/IMDB-GloVe-100dim.txt'


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

def data_integrity():
	all_ok = False
	if os.path.isdir(DATA_PREFIX):
		for part in ['train', 'test']:
			for lab in ['pos', 'neg']:
				if not os.path.isdir(os.path.join(DATA_PREFIX, which, 'pos')):
					all_ok = False
					break
			if not all_ok:
				break
	if not all_ok:
		wkdir = os.getcwd()
		os.chdir(os.path.join(DATA_PREFIX, '../'))
		import subprocess
		subprocess.call("./download.sh", shell=True)
		os.chdir(wkdir)




def get_data(positive=True, which='train'):
	_logger = logging.getLogger(__name__)
	if positive:
		examples = glob.glob(os.path.join(DATA_PREFIX, which, 'pos', '*.txt'))
	else:
		examples = glob.glob(os.path.join(DATA_PREFIX, which, 'neg', '*.txt'))
	data = []
	for i, f in enumerate(examples):
		if (i + 1) % 1000 == 0:
			log('Reading: {} of {}'.format(i + 1, len(examples)), _logger)
		data.append((open(f, 'rb').read().lower()).replace('<br /><br />', '\n'))
	return data


def parse_paragraph(txt):
	'''
	Takes a text and returns a list of lists of tokens, where each sublist is a sentence
	'''
	return [[t.text for t in s] for s in nlp(u'' + txt.decode('ascii',errors='ignore')).sents]

def parse_tokens(txt):
	'''
	Takes a text and returns a list of tokens
	'''
	return [tx for tx in (t.text for t in nlp(u'' + txt.decode('ascii',errors='ignore'))) if tx != '\n']



if __name__ == '__main__':

	log('Checking data integrity...')

	data_integrity()
	
	log('Building word vectors from {}'.format(WV_FILE))
	gb = GloVeBox(WV_FILE)
	gb.build(zero_token=True).index()

	log('writing GloVeBox pickle...')
	pickle.dump(gb, open(WV_FILE.replace('.txt', '-glovebox.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


	log('Getting training examples')
	train_neg = get_data(positive=False)
	train_pos = get_data()

	train, test = {}, {}

	log('Splitting training data into paragraphs')
	# tok_neg, tok_pos = parallel_run(parse_tokens, train_neg), parallel_run(parse_tokens, train_pos)
	train['paragraph_neg'], train['paragraph_pos'] = parallel_run(parse_paragraph, train_neg), parallel_run(parse_paragraph, train_pos)

	log('Getting testing examples')
	test_neg = get_data(positive=False, which='test')
	test_pos = get_data(which='test')

	log('Splitting testing data into paragraphs')
	# tok_neg_test, tok_pos_test = parallel_run(parse_tokens, test_neg), parallel_run(parse_tokens, test_pos)
	test['paragraph_neg'], test['paragraph_pos'] = parallel_run(parse_paragraph, test_neg), parallel_run(parse_paragraph, test_pos)


	# -- parameters to tune and set
	WORDS_PER_SENTENCE = 50
	SENTENCES_PER_PARAGRAPH = 50
 
	log('normalizing training inputs...')
	train_repr = normalize_sos(
							[
								normalize_sos(review, WORDS_PER_SENTENCE) 
								for review in gb.get_indices(train['paragraph_pos'] + train['paragraph_neg'])
							], 
			SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE
		)

	train_text = np.array(train_repr)
	train_labels = np.array([1] * len(train['paragraph_pos']) + [0] * len(train['paragraph_pos'])).astype('float32')

	log('normalizing testing inputs...')
	test_repr = normalize_sos(
						[
							normalize_sos(review, WORDS_PER_SENTENCE) 
							for review in gb.get_indices(test['paragraph_pos'] + test['paragraph_neg'])
						], 
		SENTENCES_PER_PARAGRAPH, [0] * WORDS_PER_SENTENCE
	)

	test_text = np.array(test_repr)
	test_labels = np.array([1] * len(test['paragraph_pos']) + [0] * len(test['paragraph_pos'])).astype('float32')
	log('Saving...')

	# -- training data save
	np.save('IMDB_train_glove_X.npy', train_text)
	np.save('IMDB_train_glove_y.npy', train_labels)

	# -- testing data save
	np.save('IMDB_test_glove_X.npy', test_text)
	np.save('IMDB_test_glove_y.npy', test_labels)







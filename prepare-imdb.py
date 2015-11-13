'''
prepare-imdb.py

description: prepare the imdb data for training in DNNs
'''

import os
import glob

import numpy as np
from BeautifulSoup import BeautifulSoup
from spacy.en import English

nlp = English()

DATA_PREFIX = './data/aclImdb'

def strip_html(s):
	return BeautifulSoup(s).text


def get_data(positive=True, which='train'):
	if positive:
		LABEL = 1.0
		examples = glob.glob(os.path.join(DATA_PREFIX, which, 'pos', '*.txt'))
	else:
		LABEL = 0.0
		examples = glob.glob(os.path.join(DATA_PREFIX, which, 'neg', '*.txt'))
	data = []
	for i, f in enumerate(examples):
		if (i + 1) % 100 == 0:
			print 'Reading: {} of {}'.format(i + 1, len(examples))
		data.append((open(f, 'rb').read().lower()).replace('<br /><br />', '\n'))
	return data


def parse_paragraph(txt):
	return [[t.text for t in s] for s in nlp(u'' + txt.decode('ascii',errors='ignore')).sents]

def parse_tokens(txt):
	return [t.text for t in nlp(u'' + txt.decode('ascii',errors='ignore'))]



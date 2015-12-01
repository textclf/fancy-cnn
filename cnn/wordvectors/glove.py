'''
glove.py -- a wrapper and handler for GloVe vectors from the 
GloVe package released by the Stanford NLP Group
'''

import logging

import numpy as np


LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)


try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN = True
except ImportError:
    log('Error in scikit-learn import. No indexing available.')
    SKLEARN = False

class GloVeException(Exception):
    """Errors for GloVeBox"""
    pass

class GloVeBox(object):
    '''

    A nice container for handing vocab issues and word vector lookup!

    >>> gb = GloVeBox('./glove.6B.300d.txt')
    >>> gb.build().index()
    >>> sent = ['my', 'first', 'sentence']
    >>> wv = gb[sent]
    >>> print wv.shape
    (3, 300)
    >>> 
    >>> sent = [['my', 'first', 'sentence'], ['this', 'is', 'my', 'second', 'sentence']]
    >>> ix = gb.get_indices(sent)
    >>> print ix
    [[192, 58, 2422], [37, 14, 192, 126, 2422]]


    '''
    def __init__(self, vector_file=None, verbose=True):
        super(GloVeBox, self).__init__()
        self._vector_file = vector_file
        self._verbose = verbose
        self._built = False

        self.W = None
        self.vocab = False

        self._w2i = {}
        self._i2w = {}

        self._nn = None

    def load_vectors(self, vector_file):
        self._vector_file = vector_file
        self._built = False

        self.W = None
        self.vocab = False

        self._w2i = {}
        self._i2w = {}

        self._nn = None
        return self

    def build(self, zero_token=False, normalize=True):
        if (self._vector_file is None):
            raise GloVeException('Need to specify input vector and vocab files before building')

        with open(self._vector_file, 'r') as f:
            if self._verbose:
                log('Loading vectors from {}'.format(self._vector_file))
            vectors = {}
            words = []
            ctr = 0
            for line in f:
                if ctr % 10000 == 0:
                    log('Loading word {}'.format(ctr))
                line = line.decode('utf-8')
                vals = line.rstrip().split(' ')
                if vals[0] != u'<unk>':
                    words.append(vals[0])
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                ctr += 1

        log('Building storage structures...')
        
        log('Mapping words to indices...')
        vocab_size = len(words)
        if not zero_token:
            trf = lambda x : x
        else:
            trf = lambda x : x + 1
            vocab_size += 1

        self._w2i = {unicode(w): trf(idx) for idx, w in enumerate(words)}
        self._w2i.update({'<unk>' : -1})

        if zero_token:
            self._w2i.update({'<blank>' : 0})
        
        log('Mapping indices to words...')
        self._i2w = {trf(idx): unicode(w) for idx, w in enumerate(words + ['<unk>'])}
        self._i2w.update({-1 : '<unk>'})

        if zero_token:
            self._i2w.update({0 : '<blank>'})


        vector_dim = len(vectors[self._i2w[1]])
        self.W = np.zeros((vocab_size + 1, vector_dim))
        ctr = 0

        # for word, v in vectors.iteritems():
        #     if ctr % 10000 == 0:
        #         log('Loading word vector {}'.format(ctr))
        #     if word == '<unk>':
        #         continue
        #     self.W[self._w2i[word], :] = v
        #     ctr += 1

        vs, ix = [], []
        for word, v in vectors.iteritems():
            if ctr % 10000 == 0:
                log('Loading word vector {}'.format(ctr))
            if word == '<unk>':
                continue
            vs.append(v)
            ix.append(self._w2i[word])
            ctr += 1
        self.W[np.array(ix), :] = np.array(vs)
        try:
            self.W[-1, :] = vectors['<unk>']
        except KeyError:
            self.W[-1, :] = self.W[:-1, :].mean(axis=0)

        if normalize:
            log('Normalizing vectors...')
            # normalize each word vector to unit variance

            self.W[0, :] += 1
            W_norm = np.zeros(self.W.shape)
            d = (np.sum(self.W ** 2, 1) ** (0.5))
            W_norm = (self.W.T / d).T
            self.W = W_norm
            self.W[0, :] = 0
        else:
            log('No vector normalization performed...')
        self.vocab = words
        self._built = True
        return self

    def _get_w2i(self, w):
        try:
            return self._w2i[unicode(w)]
        except KeyError:
            return self.W.shape[0] - 1

    def _get_i2w(self, i):
        try:
            return self._i2w[i]
        except KeyError:
            return '<unk>'


    def get_indices(self, obj):
        if isinstance(obj, str) or isinstance(obj, unicode):
            return self._get_w2i(obj)
        elif hasattr(obj, '__iter__'):
            return [self.get_indices(o) for o in obj]

    def get_words(self, obj):
        if isinstance(obj, int):
            return self._get_i2w(obj)
        elif hasattr(obj, '__iter__'):
            return [self.get_words(o) for o in obj]

    def __getitem__(self, key):
        if isinstance(key, unicode):
            return self.W[self._get_w2i(key), :]
        elif hasattr(key, '__iter__'):
            return self.W[np.array([self._get_w2i(k) for k in key]), :]
        elif isinstance(key, str):
            raise GloVeException('Keys must be unicode strings')

    def index(self, n_neighbors=5, metric='cosine'):
        alg = 'brute' if (metric == 'cosine') else 'auto'
        self._nn = NearestNeighbors(metric=metric, algorithm=alg)
        self._nn.fit(self.W)
        return self

    def nearest(self, word):
        '''
        Get nearest word! Can be a string or an actual word vector.

        >>> gb.nearest('sushi')
        [('sashimi', 0.2923392388266389),
         ('restaurant', 0.45604658750474103),
         ('restaurants', 0.47094956631667273),
         ('chefs', 0.4745822222385485)]
        >>>
        >>> gb.nearest(gb['dad'] - gb['man'] + gb['woman'])
        [('mom', 0.2061153295477789),
         ('dad', 0.23573104771893594),
         ('mother', 0.3477182432927921),
         ('grandmother', 0.35364849686834177),
         ('daughter', 0.42422056933288177)]

        '''
        if not SKLEARN:
            log('[WARNING] call to nearest returning None, sklearn issues present')
        if self._nn is None:
            raise GloVeException('Call to .index() necessary before queries')
        if isinstance(word, str) or isinstance(word, unicode):    
            return  [
                        (self.get_words(i), d) 
                        for d, i in zip(*
                            [
                                a.tolist()[0] for a in self._nn.kneighbors(
                                    self.__getitem__(word)
                                )
                            ]
                        ) if self.get_words(i) != word
                    ]
        else:
            return  [
                        (self.get_words(i), d) 
                        for d, i in zip(*
                            [
                                a.tolist()[0] for a in self._nn.kneighbors(word)
                            ]
                        ) if self.get_words(i) != word
                    ]









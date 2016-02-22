'''
embeddings.py -- functionality for working with non-trivial NLP embeddings
'''

from keras.layers.embeddings import Embedding
from keras.constraints import Constraint
import keras.backend as K

class ConstNorm(Constraint):
    def __init__(self, s=3, skip=True):
        self.skip = skip
        self.s = K.variable(s, name='s_constraint')
    def __call__(self, p):
        if self.skip:
            return self.s * (p / K.clip(K.sqrt(K.sum(K.square(p), axis=-1, keepdims=True)), 0.5, 100))
        return self.s * (p / K.sqrt(K.sum(K.square(p), axis=-1, keepdims=True)))

    def get_config(self):
        return {"name": self.__class__.__name__,
                "skip": self.skip,
                "s" : self.s}

def make_embedding(vocab_size, wv_size, init=None, fixed=False, constraint=ConstNorm(3.0, True), **kwargs):
    '''
    Takes parameters and makes a word vector embedding

    Args:
    ------
        vocab_size: integer -- how many words in your vocabulary

        wv_size: how big do you want the word vectors

        init: initial word vectors -- defaults to None. If you specify initial word vectors, 
                needs to be an np.array of shape (vocab_size, wv_size)

        fixed: boolean -- do you want the word vectors fixed or not?

    Returns:
    ---------

        a Keras Embedding layer
    '''
    if (init is not None) and len(init.shape) == 2:
        emb = Embedding(vocab_size, wv_size, weights=[init], W_constraint=constraint) # keras needs a list for initializations
    else:
        emb = Embedding(vocab_size, wv_size, W_constraint=constraint) # keras needs a list for initializations
    if fixed:
        emb.trainable = False
        # emb.params = []
    return emb

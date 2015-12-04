'''
embeddings.py -- functionality for working with non-trivial NLP embeddings

In particular, functionality for reshaping a 2D temporal tensor (integer lookups)
into a 5D tensor across multi-sentence embeddings.
'''

from keras.layers.containers import Graph as SubGraph
from keras.layers.core import Permute, Reshape
from keras.layers.embeddings import Embedding
from keras.constraints import Constraint
import keras.backend as K
from .version import KERAS_BACKEND

class ModifiedUnitNorm(Constraint):
    def __init__(self, skip=True):
        self.skip = skip
    def __call__(self, p):
        if self.skip:
            return K.clip(p / K.sqrt(K.sum(K.square(p), axis=-1, keepdims=True)), 1e-4, 10000)
        return p / K.sqrt(K.sum(K.square(p), axis=-1, keepdims=True))

def make_embedding(vocab_size, wv_size, init=None, fixed=False, constraint=ModifiedUnitNorm(True), **kwargs):
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

def sentence_embedding(sentence_len, wv_params, wv_size,
                       input_name='sentence_embedding', output_name='vector_embedding'):
    '''
    Creates an embedding of word vectors into a sentence image.

    Args:
    -----
        sentence_len: length of sentences to be passed

        wv_params: a dict of the following format

                        wv_params = {
                            'fixed_wv' : 
                            {
                                'vocab_size' : 1000,
                                'init' : None,
                                'fixed' : True
                            },
                            'floating_wv' : 
                            {
                                'vocab_size' : 1000,
                                'init' : None,
                                'fixed' : False
                            }
                        }
            the keys of the dictionary are the names in the keras graph model, and
            you can have any number of word vector layers encoded.

        input_name: the name of the input node for the graph

        output_name: the name of the output node for the graph

    Returns:
    --------

        a keras container that takes as input an integer array with shape (n_samples, n_words), and returns 
        shape (n_samples, wv_channels, len_sentence, wv_dim)!
    '''
    # -- output is (n_samples, n_channels, n_words, wv_dim)
    g = SubGraph()

    if KERAS_BACKEND:
        g.add_input(input_name, (sentence_len, ), dtype='int')
    else:
        g.add_input(input_name, (-1, ), dtype='int')

    for name, params in wv_params.iteritems():
        # g.add_input(params['input_name'], (-1, ), dtype='int')
        g.add_node(make_embedding(wv_size=wv_size, **params), name=name, input=input_name)

    g.add_node(Reshape((sentence_len, len(wv_params), wv_size)), name='reshape',
               inputs=wv_params.keys(), merge_mode='concat')
    g.add_node(Permute(dims=(2, 1, 3)), name='permute', input='reshape')
    
    # -- output is of shape (nb_samples, nb_wv_channels, len_sentence, wv_dim)
    g.add_output(name=output_name, input='permute')
    return g



def paragraph_embedding(paragraph_len, sentence_len, wv_params, wv_size,
        input_name='paragraph_embedding',  output_name='vector_embedding'):
    '''
    Creates an embedding of word vectors into a sequence of sentence images.

    Args:
    -----
        sentence_len: length of sentences to be passed

        wv_params: a dict of the following format

                        wv_params = {
                            'fixed_wv' : 
                            {
                                'vocab_size' : 1000,
                                'init' : None,
                                'fixed' : True
                            },
                            'floating_wv' : 
                            {
                                'vocab_size' : 1000,
                                'init' : None,
                                'fixed' : False
                            }
                        }
            the keys of the dictionary are the names in the keras graph model, and
            you can have any number of word vector layers encoded.

        input_name: the name of the input node for the graph

        output_name: the name of the output node for the graph

    Returns:
    --------

        a keras container that takes as input an integer array with shape (n_samples, n_words), where
        n_words is the number of words in the paragraph. ***NOTE*** that n_words should be organized 
        such that n_words = n_sentences * len_sentence! I.e., if you have a len_sentence of 3, and your
        input is [2, 1, 5, 0, 9, 3, 2, 4, 5], it WILL ASSUME that 

            [[2, 1, 5],
             [0, 9, 3],
             [2, 4, 5]]
        are all sentences in a paragraph.

        This returns a shape (n_samples, n_sentences, wv_channels, len_sentence, wv_dim)!
    '''
    # -- output is (n_samples, n_sentences, n_channels, n_words, wv_dim)
    g = SubGraph()
    
    # -- fix for new keras backend
    
    if KERAS_BACKEND:
        g.add_input(input_name, (sentence_len * paragraph_len, ), dtype='int')
    else:
        g.add_input(input_name, (-1, ), dtype='int')

    for name, params in wv_params.iteritems():
        # g.add_input(params['input_name'], (-1, ), dtype='int')
        g.add_node(make_embedding(wv_size=wv_size, **params), name=name, input=input_name)

    if len(wv_params.keys()) > 1:
        # -- removal for backend compatibility
        if KERAS_BACKEND:
            g.add_node(Reshape((paragraph_len, sentence_len, len(wv_params), wv_size)),
                       name='reshape', inputs=wv_params.keys(), merge_mode='concat')
        else:
            g.add_node(Reshape((-1, sentence_len, len(wv_params), wv_size)),
                name='reshape', inputs=wv_params.keys(), merge_mode='concat')

    else:
        # -- removal for backend compatibility
        if KERAS_BACKEND:
            g.add_node(Reshape((paragraph_len, sentence_len, len(wv_params), wv_size)),
                       name='reshape', input=wv_params.keys()[0])
        else:
            g.add_node(Reshape((-1, sentence_len, len(wv_params), wv_size)),
                    name='reshape', inputs=wv_params.keys(), merge_mode='concat')


    g.add_node(Permute(dims=(1, 3, 2, 4)), name='permute', input='reshape')
    
    # -- output is of shape (nb_samples, nb_wv_channels, len_sentence, wv_dim)
    g.add_output(name=output_name, input='permute')
    return g

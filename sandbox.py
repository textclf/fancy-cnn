from keras.models import Sequential
from keras.layers.containers import Sequential as Stack
from keras.layers.containers import Graph as SubGraph

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import *
from keras.layers.embeddings import Embedding

def normalize_sos(sq, sz=30):
    '''
    Take a list of lists and ensure that they are all of length `sz`

    Args:
    -----

        e: a non-generator iterable of lists


    '''
    def _normalize(e, sz):
        return e[:sz] if len(e) >= sz else e + [0] * (sz - len(e))
    return [_normalize(e, sz) for e in sq]

def sentence_embedding(sentence_len, vocab_size, wv_size, wv_init=None):
    # -- output is (n_samples, n_channels, n_words, wv_dim)
    g = SubGraph()
    
    g.add_input('sentence_emb', (-1, ), dtype='int')

    eb = Embedding(vocab_size, wv_size, weights=wv_init)
    # -- fix the word vectors.
    eb.params = []

    g.add_node(eb, name='fixed', input='sentence_emb')
    g.add_node(Embedding(vocab_size, wv_size, weights=wv_init), name='floating', input='sentence_emb')

    g.add_node(Reshape((sentence_len, 2, wv_size)), name='reshape', inputs=['fixed', 'floating'], merge_mode='concat')
    g.add_node(Permute(dims=(2, 1, 3)), name='permute', input='reshape')
    
    # -- output is of shape (nb_samples, nb_wv_channels, len_sentence, wv_dim)
    g.add_output(name='embedding', input='permute')
    return g



# -- example

x = [[1, 2, 4, 2, 5], [3, 3, 4, 7, 8], [4, 5, 21, 5], [7, 6]]
y = np.array([0, 1, 0, 1]).astype('float')

# -- sentences of len 15
d = normalize_sos(x, 15)


cnn = Sequential()

# -- len 15, 100 words, wv dim 200
g = sentence_embedding(15, 100, 200)

cnn.add(g)

cnn.add(Flatten())
cnn.add(Dropout(0.5))

cnn.add(Dense(10))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.5))

cnn.add(Dense(1))
cnn.add(Activation('sigmoid'))

cnn.compile('adam', 'binary_crossentropy')


cnn.fit(np.array(d), y, nb_epoch=2, batch_size=2)









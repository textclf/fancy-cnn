from keras.models import Sequential
from keras.layers.containers import Sequential as Stack
from keras.layers.containers import Graph as SubGraph

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import *
from keras.layers.embeddings import Embedding

from cnn.layers import embeddings
from cnn import utilities

"""
Example: 2D convolution over sentence embeddings
"""
WV_SIZE = 200
SENTENCE_LENGTH = 15

# The "text". Each element is a sentence of words
x = [[1, 2, 4, 2, 5], [3, 3, 4, 7, 8], [4, 5, 21, 5], [7, 6]]
# Labels
y = np.array([0, 1, 0, 1]).astype('float')

# Normalize to length 15
d = utilities.normalize_sos(x, SENTENCE_LENGTH)

# Use two channels, one of fixed word vectors and one of floating wvs (we train on them)
WV_PARAMS = {
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

# NN example architecture
cnn = Sequential()

g = embeddings.sentence_embedding(SENTENCE_LENGTH, WV_PARAMS, WV_SIZE)
cnn.add(g)

# 2D convolution with nb_filter=1, stride row by row,
# reading the whole word vector in a single stride
conv2d = Convolution2D(1, 1, WV_SIZE, subsample=(1,1))
cnn.add(conv2d)

cnn.add(Flatten())
cnn.add(Dropout(0.5))

cnn.add(Dense(10))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.5))

cnn.add(Dense(1))
cnn.add(Activation('sigmoid'))

cnn.compile('adam', 'binary_crossentropy')
cnn.fit(np.array(d), y, nb_epoch=2, batch_size=2)
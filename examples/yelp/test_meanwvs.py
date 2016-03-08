import cPickle as pickle
from os.path import join as path_join
import sys
import numpy as np

from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Permute
from keras.layers import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import MaxoutDense, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

ROOT_PATH = '../..'
sys.path.append(ROOT_PATH)

from textclf.nn import train_neural
from textclf.wordvectors import misc as wvmisc

MODEL_FILE = './models/funny/yelp-model-mean-vws-1'
LOG_FILE = './outputs/funny/log-model-mean-vws-1'

## Where is the data
train_reviews = np.load("../../Yelp_funny_train_fulltext_glove_300_X.npy")
train_labels = np.load("../../Yelp_funny_train_fulltext_glove_300_y.npy")
test_reviews = np.load("../../Yelp_funny_test_fulltext_glove_300_X.npy")
test_labels = np.load("../../Yelp_funny_test_fulltext_glove_300_y.npy")

# train_reviews = np.load("../../Yelp_useful_train_fulltext_glove_300_X.npy")
# train_labels = np.load("../../Yelp_useful_train_fulltext_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_useful_test_fulltext_glove_300_X.npy")
# test_labels = np.load("../../Yelp_useful_test_fulltext_glove_300_y.npy")

# train_reviews = np.load("../../Yelp_cool_train_fulltext_glove_300_X.npy")
# train_labels = np.load("../../Yelp_cool_train_fulltext_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_cool_test_fulltext_glove_300_X.npy")
# test_labels = np.load("../../Yelp_cool_test_fulltext_glove_300_y.npy")

### Same with Yelp glove vectors
#
# train_reviews = np.load("../../Yelp_funny_train_fulltext_Yelp_glove_300_X.npy")
# train_labels = np.load("../../Yelp_funny_train_fulltext_Yelp_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_funny_test_fulltext_Yelp_glove_300_X.npy")
# test_labels = np.load("../../Yelp_funny_test_fulltext_Yelp_glove_300_y.npy")

# train_reviews = np.load("../../Yelp_useful_train_fulltext_Yelp_glove_300_X.npy")
# train_labels = np.load("../../Yelp_useful_train_fulltext_Yelp_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_useful_test_fulltext_Yelp_glove_300_X.npy")
# test_labels = np.load("../../Yelp_useful_test_fulltext_Yelp_glove_300_y.npy")
#
# train_reviews = np.load("../../Yelp_cool_train_fulltext_Yelp_glove_300_X.npy")
# train_labels = np.load("../../Yelp_cool_train_fulltext_Yelp_glove_300_y.npy")
# test_reviews = np.load("../../Yelp_cool_test_fulltext_Yelp_glove_300_X.npy")
# test_labels = np.load("../../Yelp_cool_test_fulltext_Yelp_glove_300_y.npy")

#WV_FILE_GLOBAL = path_join(ROOT_PATH, 'embeddings/wv/glove.42B.300d.120000-glovebox.pkl')
WV_FILE_GLOBAL = path_join(ROOT_PATH, 'embeddings/wv/Yelp-GloVe-300dim-glovebox.pkl')

gb = pickle.load(open(WV_FILE_GLOBAL, 'rb'))

print "Computing mean vectors"
X_train = wvmisc.data_to_wvs(gb, train_reviews)
X_test = wvmisc.data_to_wvs(gb, test_reviews)

np.save("funny_wv_means_train.npy", X_train)
np.save("funny_wv_means_test.npy", X_test)

#X_train = np.load("funny_wv_means_train.npy")
#X_test = np.load("funny_wv_means_test.npy")



model = Sequential()

#model.add(MaxoutDense(100, 20, input_shape=(X_train.shape[1],))) # SALE 100
model.add(Dense(100, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#model.add(MaxoutDense(20, 10)) # SALE 20
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))


#model.add(MaxoutDense(1, 1)) # SALE 1
model.add(Dense(5))
model.add(Activation('relu'))
# model_basic.add(Dropout(0.1))

model.add(Dense(1))
model.add(Activation('tanh'))

# model_basic.add(Dense(10, 1))
# model_basic.add(Activation('relu'))

model.compile(loss='binary_crossentropy', optimizer="adam", class_mode="binary")


history = train_neural.train_sequential(model, X_train, train_labels, MODEL_FILE)
acc = train_neural.test_sequential(model, X_test, test_labels, MODEL_FILE)
train_neural.write_log(model, history, __file__, acc, LOG_FILE)

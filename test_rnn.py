import numpy as np
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, model_from_json, Graph
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD
from keras.regularizers import l2

import cPickle as pickle

from cnn.layers.embeddings import *

MODEL_FILE = './imdb-model-gru'

# Read back data
train_reviews = np.load("IMDB_train_fulltext_glove_X.npy")
# train_reviews_2 = []
# for review in train_reviews:
#     train_reviews_2.append(np.array(review))
# train_reviews = np.array(train_reviews_2)

train_labels = np.load("IMDB_train_fulltext_glove_y.npy")

test_reviews = np.load("IMDB_test_fulltext_glove_X.npy")
test_labels = np.load("IMDB_test_fulltext_glove_y.npy")

WV_FILE_GLOBAL = './data/wv/glove.42B.300d.120000-glovebox.pkl'
gb_global = pickle.load(open(WV_FILE_GLOBAL, 'rb'))

wv_size = gb_global.W.shape[1]

model = Sequential()
model.add(make_embedding(vocab_size=gb_global.W.shape[0], init=gb_global.W, wv_size=wv_size,
                         fixed=True, constraint=None))
model.add(GRU(128, init='uniform'))
#model.add(Dropout(0.2))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

# print("Train...")
# model.fit(train_reviews, train_labels,
#           batch_size=16, nb_epoch=15, validation_split=0.1, show_accuracy=True)

print 'Fitting! Hit CTRL-C to stop early...'
try:
    history = model.fit(train_reviews,
                        train_labels,
                        batch_size=32,
                        nb_epoch=15,
                        verbose=True,
                        validation_split=0.1,
                        callbacks=[
                            EarlyStopping(verbose=True, patience=30, monitor='val_loss'),
                            ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=True, save_best_only=True)
                        ])
except KeyboardInterrupt:
    print "Training stopped early!"

print "Loading best weights..."
model.load_weights(MODEL_FILE)

print "getting predictions on the test set"
yhat = model.predict(test_reviews, verbose=True, batch_size=50)
acc = ((yhat.ravel() > 0.5) == (test_labels > 0.5)).mean()

print "Test set accuracy of {}%.".format(acc * 100.0)
print "Test set error of {}%. Exiting...".format((1 - acc) * 100.0)
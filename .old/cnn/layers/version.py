try:
    from keras.backend import floatx as _
    KERAS_BACKEND = True
except ImportError:
    KERAS_BACKEND = False
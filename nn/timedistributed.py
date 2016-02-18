import numpy as np

from collections import OrderedDict
import copy
from six.moves import zip

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer
from keras.layers.core import MaskedLayer


class TimeDistributed(MaskedLayer):
    def __init__(self, layer, input_shape=None, input_dim=None, input_length=None, weights=None, **kwargs):
        self.layer = layer

        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length

        if hasattr(self.layer, 'input_ndim'):
            self.input_ndim = self.layer.input_ndim + 1

        if input_shape:
            self.set_input_shape((None, ) + input_shape)

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(TimeDistributed, self).__init__(**kwargs)

    def set_previous(self, layer):
        self.input_ndim = len(layer.output_shape)
        super(TimeDistributed, self).set_previous(layer)

    def build(self):
        try:
            self.input_ndim = len(self.previous.input_shape)
        except AttributeError:
            self.input_ndim = len(self.input_shape)

        self.layer.set_input_shape((None, ) + self.input_shape[2:])

        if hasattr(self.layer, 'regularizers'):
            self.regularizers = self.layer.regularizers

        if hasattr(self.layer, 'constraints'):
            self.constraints = self.layer.constraints
        
        if hasattr(self.layer, 'trainable_weights'):
            self.trainable_weights = self.layer.trainable_weights

            if self.initial_weights is not None:
                self.layer.set_weights(self.initial_weights)
                del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1]) + self.layer.output_shape[1:]

    def get_output(self, train=False):
        def format_shape(shape):
            if K._BACKEND == 'tensorflow':
                return map(int, shape)
            return shape

        X = self.get_input(train)

        in_shape = format_shape(K.shape(X))
        batch_flatten_len = K.prod(in_shape[:2])
        cast_in_shape = (batch_flatten_len, ) + tuple(in_shape[i] for i in range(2, K.ndim(X)))
        
        pre_outs = self.layer(K.reshape(X, cast_in_shape))
        
        out_shape = format_shape(K.shape(pre_outs))
        cast_out_shape = (in_shape[0], in_shape[1]) + tuple(out_shape[i] for i in range(1, K.ndim(pre_outs)))
        
        outputs = K.reshape(pre_outs, cast_out_shape)
        return outputs

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layer': self.layer.get_config(),
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(TimeDistributed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
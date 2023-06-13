import tensorflow as tf
import tensorflow.keras as keras
from time import time

K = keras.backend

from util.Layers import *

def GarNetModel(aggregators=([4, 4, 8]), filters=([8, 8, 16]), propagate=([8, 8, 16]), 
                input_format='xn', vmax=128, simplified=True, collapse='mean', quantize=False):
    """ """
    x = keras.layers.Input(shape=(vmax, 4))
    n = keras.layers.Input(shape=(1,), dtype='uint16')
    
    if input_format == 'xn': 
        inputs = [x, n]
    elif input_format == 'x':
        inputs = x
    else:
        raise ValueError(f'input_format must be one of [\'x\', \'xn\'] not {input_format}')
        
    v = GarNetStack(aggregators, filters, propagate, 
                    simplified=simplified, collapse=collapse, 
                    input_format=input_format, output_activation=None, name='garnet', quantize_transforms=quantize)(inputs)
    v = keras.layers.Dense(16, activation='relu')(v)
    v = keras.layers.Dense(8, activation='relu')(v)
    b = keras.layers.Dense(2, activation='sigmoid', name='classification')(v)
    p = keras.layers.Dense(1, name='regression')(v)
    outputs = [b, p]

    return keras.Model(inputs=inputs, outputs=outputs)

class PrinterCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time()
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        prefix = f'Epoch {self.epoch}/{self.params["epochs"]}: '
        size = 50
        count = self.params["steps"]
        x = int(size*batch/count)

        print(f"{prefix}[{'='*x}{('.'*(size-x))}] {batch}/{count}", end='\r', flush=True)
        
    def on_epoch_end(self, epoch, logs=None):
        self.end = time()
        prefix = f'Epoch {epoch}/{self.params["epochs"]}: '
        size = 50
        count = self.params["steps"]
        print(f"{prefix}[{'='*size}] {count}/{count}")
        print(f'{int(self.end - self.start):2d}s - loss: {logs["loss"]:.4f} - val loss: {logs["val_loss"]:.4f}')


class GarNetClusteringModel(keras.Model):
    """ """
    def __init__(self, aggregators=([4, 4, 8]), filters=([8, 8, 16]), propagate=([8, 8, 16]), **kwargs):
        """ """
        super().__init__(**kwargs)
        
        self.aggregators = aggregators
        self.filters = filters
        self.propagate = propagate
        
        self.blocks = []
        
        block_params = zip(aggregators, filters, propagate)
        
        momentum = kwargs.get('momentum', 0.99)
        self.input_gex = self.add_layer(GlobalExchange, name='input_gex')
        self.input_batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='input_batchnorm')
        self.input_dense = self.add_layer(keras.layers.Dense, 8, activation='tanh', name='input_dense')
        
        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self.add_layer(GarNet, normalizer, n_aggregators, n_filters, n_propagate, name='garnet_%d' % i)
            batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm_%d' % i)
            
            self.blocks.append((garnet, batchnorm))
        
        self.output_dense_0 = self.add_layer(keras.layers.Dense, 16, activation='relu', name='output_0')
        self.output_dense_1 = self.add_layer(keras.layers.Dense, 8, activation='relu', name='output_1')
        self.output_classification = self.add_layer(keras.layers.Dense, 2, activation='sigmoid', name='classification')
        self.output_regression = self.add_layer(keras.layers.Dense, 1, name='regression')
        
    def call(self, inputs):
        """ """
        features = []
        
        x = inputs
        x = self.input_gex(x)
        x = self.input_batchnorm(x)
        x = self.input_dense(x)
        
        for block in self.blocks:
            for layer in block:
                x = layer(x)
            features.append(x)
        
        x = tf.concat(features, axis=-1)
        
        x = K.mean(x, axis=-2)
        x = self.output_dense_0(x)
        x = self.output_dense_1(x)
        
        b = self.output_classification(x)
        p = self.output_regression(x)
        
        return K.concatenate([b, p], axis=-1)
    
    def add_layer(self, cls, *args, **kwargs):
        """ """
        layer = cls(*args, **kwargs)
        self.layers.append(layer)
        return layer
    
    def get_config(self):
        config = {'aggregators':self.aggregators,
                  'filters':self.filters,
                  'propagate':self.propagate}
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import hls4ml
from time import time

K = keras.backend

from PionReconstruction.util.Layers import *

def GarNetModel(aggregators=([4, 4, 8]), filters=([8, 8, 16]), propagate=([8, 8, 16]), input_format='xne', vmax=128, simplified=True, collapse='mean', mean_by_nvert=True, quantize=False):
    """ """
    layer1 = keras.layers.Dense(16, activation='relu', name='dense_1')
    layer2 = keras.layers.Dense(8, activation='relu', name='dense_2')
    classification = keras.layers.Dense(2, activation='sigmoid', name='classification')
    regression = keras.layers.Dense(1, activation='linear', name='regression')
    
    
    x = keras.layers.Input(shape=(vmax, 4), name='data')
    n = keras.layers.Input(shape=(1,), name='vertex', dtype='uint16')
    e = keras.layers.Input(shape=(1,), name='energy')
    
    if input_format == 'xne':
        inputs = [x, n, e]
        def format_inputs(xne):
            return xne[0:2]
    elif input_format == 'xn':
        inputs = [x, n]
        def format_inputs(xn):
            return xn
    else:
        raise NotImplementedError(f'GarNet only supports "xne" and "xn" input formats, not {input_format}')
        
    

    v = GarNetStack(aggregators, filters, propagate, 
                    simplified=simplified, mean_by_nvert=mean_by_nvert, collapse=collapse, 
                    output_activation=None, name='garnet', quantize_transforms=quantize)(format_inputs(inputs))
    if input_format == 'xne':
        v = keras.layers.Concatenate()([v, inputs[-1]])
    v = layer1(v)
    v = layer2(v)
    b = classification(v)
    p = regression(v)
    outputs = [b, p]

    return keras.Model(inputs=inputs, outputs=outputs)

def keras_weights(model):
    """ """
    names = []
    weights = []
    for layer in model.layers:
        for idx, weight in enumerate(layer.get_weights()):
            name = layer.weights[idx].name[:-2]
            if name.endswith('kernel'):
                name = name[:-7].replace('garnet/','') + '/w'
            if name.endswith('bias'):
                name = name[:-5].replace('garnet/','') + '/b'
            names.append(name)
            
            w = weight.flatten()
            w = np.abs(w[w != 0])
            weights.append(w)

    return names, weights

def keras_activations(model, x):
    """ """
    activations = []
    names = []
    for layer in model.layers:
        if not isinstance(layer, keras.layers.InputLayer):
            try:
                for sub in layer._sublayers:
                    tmp_model = keras.models.Model(inputs=model.input, outputs=sub.output)
                    y = tmp_model.predict(x).flatten()
                    y = np.abs(y[y != 0])
                    activations.append(y)
                    names.append(sub.name)
            except:
                tmp_model = keras.models.Model(inputs=model.input, outputs=layer.output)
                y = tmp_model.predict(x).flatten()
                y = np.abs(y[y != 0])
                activations.append(y)
                names.append(layer.name)

    return names, activations

def hls_weights(hls_model):
    names = []
    weights = []

    for layer in hls_model.get_layers():
        for weight in layer.get_weights():
            name = weight.name
            if 'input_transform_' in name:
                name = name.replace('input_transform_', 'FLR')[:-1]
                name = name.replace('_', '/')
            elif 'aggregator_distance_' in name:
                name = name.replace('aggregator_distance_', 'S')[:-1]
                name = name.replace('_', '/')
            elif 'output_transform_' in name:
                name = name.replace('output_transform_', 'Fout')[:-1]
                name = name.replace('_', '/')
            else:
                name = layer.name + '/' + weight.name[0]
            
            names.append(name)
            
            w = weight.data.flatten()
            w = np.abs(w[w != 0])
            weights.append(w)

    return names, weights

def hls_activations(hls_model, x):
    names = []
    activations = []
    
    _, trace = hls_model.trace(x)

    for layer in trace.keys():
        y = trace[layer].flatten()
        y = abs(y[y != 0])
        activations.append(y)
        names.append(layer)

    return names, activations

def weight_types(hls_model):
    """ """
    precisions = {'layer': [], 'low': [], 'high': []}
    for layer in hls_model.get_layers():
        for weight in layer.get_weights():
            name = weight.name
            if 'input_transform_' in name:
                name = name.replace('input_transform_', 'FLR')[:-1]
                name = name.replace('_', '/')
            elif 'aggregator_distance_' in name:
                name = name.replace('aggregator_distance_', 'S')[:-1]
                name = name.replace('_', '/')
            elif 'output_transform_' in name:
                name = name.replace('output_transform_', 'Fout')[:-1]
                name = name.replace('_', '/')
            else:
                name = layer.name + '/' + weight.name[0]
            t = weight.type
            W, I, F, S = hls4ml.model.profiling.ap_fixed_WIFS(t.precision)
            precisions['layer'].append(name)
            precisions['low'].append(-F)
            precisions['high'].append(I - 1 if S else I)
    
    return precisions

def activation_types(hls_model):
    """ """
    precisions = {'layer': [], 'low': [], 'high': []}
    for layer in hls_model.get_layers():
        t = layer.get_output_variable().type
        W, I, F, S = hls4ml.model.profiling.ap_fixed_WIFS(t.precision)
        precisions['layer'].append(layer.name)
        precisions['low'].append(-F)
        precisions['high'].append(I - 1 if S else I)
    
    return precisions

class PrinterCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time()
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        prefix = f'Epoch {self.epoch + 1}/{self.params["epochs"]}: '
        size = 50
        count = self.params["steps"]
        x = int(size*(batch + 1)/count)

        print(f"{prefix}[{'='*x}{('.'*(size-x))}] {batch + 1}/{count}", end='\r', flush=True)
        
    def on_epoch_end(self, epoch, logs=None):
        self.end = time()
        prefix = f'Epoch {epoch + 1}/{self.params["epochs"]}: '
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

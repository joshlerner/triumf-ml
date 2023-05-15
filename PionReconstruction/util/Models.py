import tensorflow as tf
import tensorflow.keras as keras

K = keras.backend

from util.Layers import *

class GarNetModel(keras.Model):
    """ """
    def __init__(self, aggregators=([4]*11), filters=([32]*11), propagate=([20]*11), **kwargs):
        """ """
        super(GarNetModel, self).__init__(**kwargs)
        
        self.blocks = []
        
        block_params = zip(aggregators, filters, propagate)
        
        momentum = kwargs.get('momentum', 0.99)
        self.input_gex = self.add_layer(GlobalExchange, name='input_gex')
        self.input_batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='input_batchnorm')
        self.input_dense = self.add_layer(keras.layers.Dense, 32, activation='tanh', name='input_dense')
        
        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self.add_layer(GarNet, n_aggregators, n_filters, n_propagate, name='garnet_%d' % i)
            batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm_%d' % i)
            
            self.blocks.append((garnet, batchnorm))
            
        self.output_dense_0 = self.add_layer(keras.layers.Dense, 48, activation='relu', name='output_0')
        self.output_dense_1 = self.add_layer(keras.layers.Dense, 3, activation='relu', name='output_1')
        
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
        
        x = self.output_dense_0(x)
        x = self.output_dense_1(x)
        
        return x
    
    def add_layer(self, cls, *args, **kwargs):
        """ """
        layer = cls(*args, **kwargs)
        self.layers.append(layer)
        return layer
    
    def summary(self):
        inputs = keras.Input(shape=(128, 4,))
        outputs = self.call(inputs)
        keras.Model(inputs=inputs, outputs=outputs, name=self.name).summary()
        
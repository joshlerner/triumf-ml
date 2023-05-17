import tensorflow as tf
import tensorflow.keras as keras

K = keras.backend

from util.Layers import *

class GarNetModel(keras.Model):
    """ """
    def __init__(self, aggregators=([4, 4, 8]), filters=([8, 8, 16]), propagate=([8, 8, 16]), summarize=True, **kwargs):
        """ """
        super(GarNetModel, self).__init__(**kwargs)
        
        self.blocks = []
        
        block_params = zip(aggregators, filters, propagate)
        
        momentum = kwargs.get('momentum', 0.99)
        self.input_gex = self.add_layer(GlobalExchange, name='input_gex')
        self.input_batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='input_batchnorm')
        self.input_dense = self.add_layer(keras.layers.Dense, 8, activation='tanh', name='input_dense')
        
        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self.add_layer(GarNet, n_aggregators, n_filters, n_propagate, name='garnet_%d' % i)
            batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm_%d' % i)
            
            self.blocks.append((garnet, batchnorm))
        
        self.output_dense_0 = self.add_layer(keras.layers.Dense, 16, activation='relu', name='output_0')
        self.output_dense_1 = self.add_layer(keras.layers.Dense, 8, activation='relu', name='output_1')
        self.output_classification = self.add_layer(keras.layers.Dense, 2, activation='sigmoid', name='classification')
        self.output_regression = self.add_layer(keras.layers.Dense, 1, name='regression')
        
        self.compile(loss=self.loss_fcn, optimizer='adam')
        
        if summarize:
            self.summary()
        
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
        
        #x = tf.concat(features, axis=-1) What does this do?
        
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
    
    def summary(self):
        """ """
        inputs = keras.Input(shape=(128, 4,))
        outputs = self.call(inputs)
        keras.Model(inputs=inputs, outputs=outputs, name=self.name).summary() 
        
    
    def loss_fcn(y_true, y_pred):
        """ """
        bce = keras.losses.BinaryCrossentropy()
        mse = K.mean(K.square((y_true[:,2:3] - y_pred[:,2:3]) / y_true[:,2:3]), axis=-1)
        
        return 0.01*bce(y_true[:,0:2], y_pred[:,0:2]) + 0.99*mse
    
        
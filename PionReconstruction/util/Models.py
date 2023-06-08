import tensorflow as tf
import tensorflow.keras as keras

K = keras.backend

from util.Layers import *

def scheduler(epoch, lr):
    if epoch < 15:
        return lr
    else:
        return lr*tf.math.exp(-0.03)

class GarNetModel(keras.Model):
    """ """
    def __init__(self, alpha=0.50, normalizer='log', aggregators=([4, 4, 8]), filters=([8, 8, 16]), propagate=([8, 8, 16]), summarize=False, **kwargs):
        """ """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.normalizer = normalizer
        self.aggregators = aggregators
        self.filters = filters
        self.propagate = propagate
        self.summarize = summarize
        self.kwargs = kwargs
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
        
        def loss_fcn(y_true, y_pred):
            bce = keras.losses.BinaryCrossentropy()
            if normalizer is not None:
                mse = keras.losses.MeanSquaredError()
            else:
                def mse(ytrue, ypred):
                    return ((y_true - y_pred)/(y_true + 0.001))**2
            return alpha*bce(y_true[:,0:2], y_pred[:,0:2]) + (1-alpha)*mse(y_true[:,2:3], y_pred[:,2:3])
        
        self.compile(loss=loss_fcn, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
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
    
    def summary(self):
        """ """
        inputs = keras.Input(shape=(128, 4,))
        outputs = self.call(inputs)
        keras.Model(inputs=inputs, outputs=outputs, name=self.name).summary() 
    
    def get_config(self):
        return {'alpha':self.alpha,
                'normalizer':self.normalizer,
                'aggregators':self.aggregators,
                'filters':self.filters,
                'propagate':self.propagate,
                'summarize':self.summarize}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

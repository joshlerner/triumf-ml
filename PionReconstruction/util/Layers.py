import tensorflow as tf
import tensorflow.keras as keras

K = keras.backend

try:
    from qkeras import QDense, ternary

    class NamedQDense(QDense):
        """ A special variation of the QDense layer class with assignable name """
        def add_weight(self, name=None, **kwargs):
            return super(NamedQDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

    def quantizer():
        return ternary(alpha=1., threshold=0.5)

except ImportError:
    pass

class NamedDense(keras.layers.Dense):
    """ A special variation of the Dense layer class with assignable name """
    def add_weight(self, name=None, **kwargs):
        return super(NamedDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

class GarNet(keras.layers.Layer):
    """ 
    The GarNet layer class for aggregating vertex features and passing back a weighted message
    
    Modified from `caloGraphNN_keras.py` found at <https://github.com/jkiesele/caloGraphNN>
    
    For more information on GarNet, see the following paper at <arXiv:1902.07987>
    
    Qasim, S., Kieseler, J., Iiyama, Y., & Pierini, M. (2019). Learning representations of irregular particle-detector geometry with distance-weighted graph networks. Eur. Phys. J., C79(7), 608.

    
    Attributes
    ----------
    output_activation : str
        the activation used to decode the aggregated features
    input_format : str
        the data format of the input data as specified by the generator
    quantize_transforms : bool
        if the encoder and decoder are quantized during training
    simplified : bool
        if the layer is simplified by aggregating only the mean of the learned features
    collapse : bool
        if the output tensor is collapsed along the outermost dimension
    mean_by_nvert : bool
        if means are are calculated from the number of non-zero vertices
    
    """
    def __init__(self, n_aggregators, n_filters, n_propagate,
                 output_activation='linear',
                 quantize_transforms=False,
                 simplified=True,
                 mean_by_nvert=False,
                 collapse=None,
                 input_format='xn',
                 **kwargs):
        """ Initialization """
        
        super().__init__(**kwargs)
        
        self._simplified = simplified
        self._quantize_transforms = quantize_transforms
        self._output_activation=output_activation
        self._collapse = collapse
        self._mean_by_nvert = mean_by_nvert
        self._input_format = input_format
        
        self._setup_transforms(n_aggregators, n_filters, n_propagate)
        
    def _setup_transforms(self, n_aggregators, n_filters, n_propagate):
        """ Create encoder and decoder layers """
        if self._quantize_transforms:
            self._input_feature_transform = NamedQDense(n_propagate, 
                                                        kernel_quantizer=quantizer(),
                                                        bias_quantizer=quantizer(),
                                                        name='FLR')
            self._output_feature_transform = NamedQDense(n_filters, 
                                                         activation=self._output_activation,
                                                         kernel_quantizer=quantizer(),
                                                         name='Fout')
        else:
            self._input_feature_transform = NamedDense(n_propagate, name='FLR')
            self._output_feature_transform = NamedDense(n_filters, 
                                                        activation=self._output_activation, 
                                                        name='Fout')
        self._aggregator_distance = NamedDense(n_aggregators, name='S')
        self._sublayers = [self._input_feature_transform, self._aggregator_distance, self._output_feature_transform]

    def build(self, input_shape):
        """ Build GarNet layer by specifying data shapes of consecutive sublayers """
        if self._input_format == 'x':
            data_shape = input_shape
        elif self._input_format == 'xn':
            data_shape, _ = input_shape
            
        self._build_transforms(data_shape)
        
        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)
        
    def _build_transforms(self, data_shape):
        """ Build encoder and decoder layers by computing expected data shapes """
        self._input_feature_transform.build(data_shape)
            
        if self._simplified:
            self._output_feature_transform.build(data_shape[:2] + \
                                                 (self._aggregator_distance.units * \
                                                  self._input_feature_transform.units,))
        else:
            self._output_feature_transform.build(data_shape[:2] + \
                                                 (data_shape[2] + d_compute.units * \
                                                  2 * (d_compute.units + in_transform.units) + \
                                                  d_compute.units,))

    def call(self, x):
        """ Data flow of sublayers in GarNet layer """
        data, num_vertex, vertex_mask = self._unpack_input(x)
        
        output = self._garnet(data, num_vertex, vertex_mask,
                              self._input_feature_transform,
                              self._aggregator_distance,
                              self._output_feature_transform)
        
        output = self._collapse_output(output, num_vertex)
        
        return output
    
    def _unpack_input(self, x):
        """ Extract vertices, features, and mask from data """
        data, num_vertex = x # 'xn' is the only HLS supported format
        
        data_shape = K.shape(data)
        B = data_shape[0]
        V = data_shape[1]
        vertex_indices = K.tile(K.expand_dims(K.arange(0, V), axis=0), (B, 1))
        vertex_mask = K.expand_dims(K.cast(K.less(vertex_indices, K.cast(num_vertex, 'int32')), 'float32'), axis=-1)
        num_vertex = K.cast(num_vertex, 'float32')
        
        return data, num_vertex, vertex_mask
    
    def _garnet(self, data, num_vertex, vertex_mask, in_transform, d_compute, out_transform):
        """ Perform encoding, aggregation, message propagation, and decoding layer operations """
        features = in_transform(data) # (B, V, F)
        distance = d_compute(data) # (B, V, S)
        edge_weights = vertex_mask * K.exp(-K.square(distance)) # (B, V, S)
        
        if not self._simplified:
            features = K.concatenate([vertex_mask * features, edge_weights], axis=-1) # F = F + S
            
        if self._mean_by_nvert:
            if self._simplified:
                def graph_mean(out, axis):
                    s = K.sum(out, axis)
                    s = K.reshape(s, (-1, d_compute.units * in_transform.units)) / num_vertex
                    s = K.reshape(s, (-1, d_compute.units, in_transform.units))
                    return s
            else:
                def graph_mean(out, axis):
                    s = K.sum(out, axis=axis)
                    s = K.reshape(s, (-1, d_compute.units * (d_compute.units + in_transform.units))) / num_vertex
                    s = K.reshape(s, (-1, d_compute.units, (d_compute.units + in_transform.units)))
                    return s
        else:
            graph_mean = K.mean
            
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)

        aggregated_mean = self._apply_edge_weights(features, edge_weights_trans, aggregation=graph_mean) # (B, S, F)

        
        if self._simplified:
            aggregated = aggregated_mean
        else:
            aggregated_max = self._apply_edge_weights(features, edge_weights_trans, aggregation=K.max)
            aggregated = K.concatenate([aggregated_max, aggregated_mean], axis=-1) # F = 2*F

        updated_features = self._apply_edge_weights(aggregated, edge_weights) # (B, V, S*F)
        
        if not self._simplified:
            updated_features = K.concatenate([data, updated_features, edge_weights], axis=-1) # F = F + D + data[2]
        updated_features = vertex_mask * out_transform(updated_features)
        
        return updated_features

    def _collapse_output(self, output, num_vertex):
        """ Collapse output tensor along outermost dimension """
        if self._collapse == 'mean':
            if self._mean_by_nvert:
                output = K.sum(output, axis=1) / num_vertex
            else:
                output = K.mean(output, axis=1)
        elif self._collapse == 'sum':
            output = K.sum(output, axis=1)
        elif self._collapse == 'max':
            output = K.max(output, axis=1)
            
        return output
    
    def compute_output_shape(self, input_shape):
        """ Retrieve the output shape for the given input shape"""
        return self._get_output_shape(input_shape, self._output_feature_transform)
    
    def _get_output_shape(self, input_shape, output_transform):
        """ Determine the output shape from the given input shape and output transform """
        data_shape = input_shape
            
        if self._collapse is None:
            return data_shape[:2] + (out_transform.units,)
        else:
            return (data_shape[0], out_transform.units)
        
    def get_config(self):
        """ Return an custom layer config for the GarNet layer """
        config = super().get_config()
        
        config.update({
            'simplified': self._simplified,
            'collapse':self._collapse,
            'input_format':self._input_format,
            'output_activation':self._output_activation,
            'quantize_transforms':self._quantize_transforms,
            'mean_by_nvert':self._mean_by_nvert})
        self._add_transform_config(config)
        
        return config
    
    def _add_transform_config(self, config):
        """ Update the config with the number of aggregators, filters, and propagators in the layer """
        config.update({
            'n_aggregators':self._aggregator_distance.units,
            'n_filters':self._output_feature_transform.units,
            'n_propagate':self._input_feature_transform.units})
    
    @staticmethod
    def _apply_edge_weights(features, edge_weights, aggregation=None):
        """ Application of edge weights for aggregation and message passing """
        features = K.expand_dims(features, axis=1) # (B, 1, V, F)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, S, V, 1)
        out = edge_weights * features # (B, S, V, F)
        
        if aggregation:
            out = aggregation(out, axis=2)
        else:
            out = K.reshape(out, (-1, edge_weights.shape[1], features.shape[-1] * features.shape[-2]))
        return out

class GarNetStack(GarNet):
    """
    Stacked version of GarNet 
    First three arguments to the constructor must be lists of integers
    Basically offers no performance advantage, but the configuration is consolidated 

    """
    
    def _setup_transforms(self, n_aggregators, n_filters, n_propagate):
        self._transform_layers = []
        for it, (p, a, f) in enumerate(zip(n_propagate, n_aggregators, n_filters)):
            if self._quantize_transforms:
                input_feature_transform = NamedQDense(p, kernel_quantizer=quantizer(), bias_quantizer=quantizer(), name=('FLR%d' % it))
                output_feature_transform = NamedQDense(f, activation=self._output_activation, kernel_quantizer=quantizer(), name=('Fout%d' % it))
            else:
                input_feature_transform = NamedDense(p, name=('FLR%d' % it))
                output_feature_transform = NamedDense(f, activation=self._output_activation, name=('Fout%d' % it))

            aggregator_distance = NamedDense(a, name=('S%d' % it))

            self._transform_layers.append((input_feature_transform, aggregator_distance, output_feature_transform))

        self._sublayers = sum((list(layers) for layers in self._transform_layers), [])

    def _build_transforms(self, data_shape):
        for in_transform, d_compute, out_transform in self._transform_layers:
            in_transform.build(data_shape)
            d_compute.build(data_shape)
            if self._simplified:
                out_transform.build(data_shape[:2] + \
                                    (d_compute.units * in_transform.units,))
            else:
                out_transform.build(data_shape[:2] + \
                                    (data_shape[2] + d_compute.units * \
                                     2 * (d_compute.units + in_transform.units) + \
                                     d_compute.units,))

            data_shape = data_shape[:2] + (out_transform.units,)

    def call(self, x):
        data, num_vertex, vertex_mask = self._unpack_input(x)

        for in_transform, d_compute, out_transform in self._transform_layers:
            data = self._garnet(data, num_vertex, vertex_mask, in_transform, d_compute, out_transform)
    
        output = self._collapse_output(data, num_vertex)

        return output

    def compute_output_shape(self, input_shape):
        return self._get_output_shape(input_shape, self._transform_layers[-1][2])

    def _add_transform_config(self, config):
        config.update({
            'n_propagate': list(ll[0].units for ll in self._transform_layers),
            'n_aggregators': list(ll[1].units for ll in self._transform_layers),
            'n_filters': list(ll[2].units for ll in self._transform_layers),
            'n_sublayers': len(self._transform_layers)})
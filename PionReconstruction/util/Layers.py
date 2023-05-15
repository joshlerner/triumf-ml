import tensorflow.keras as keras

K = keras.backend

try:
    from qkeras import QDense, ternary

    class NamedQDense(QDense):
        def add_weight(self, name=None, **kwargs):
            return super(NamedQDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

    def ternary_1_05():
        return ternary(alpha=1., threshold=0.5)
except ImportError:
    pass

class NamedDense(keras.layers.Dense):
    def add_weight(self, name=None, **kwargs):
        return super(NamedDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)
    
class GlobalExchange(keras.layers.Layer):
    """ """
    def __init__(self, vertex_mask=None, **kwargs):
        """ """
        super().__init__(**kwargs)
        
        self.vertex_mask = vertex_mask
        
    def build(self, input_shape):
        """ """
        self.num_vertices = input_shape[1]
        super().build(input_shape)
        
    def call(self, x):
        """ """
        mean = K.mean(x, axis=1, keepdims=True)
        mean = K.tile(mean, [1, self.num_vertices, 1])
        if self.vertex_mask is not None:
            mean = self.vertex_mask * mean
        return K.concatenate([x, mean], axis=-1)
    
    def compute_output_shape(self, input_shape):
        """ """
        return input_shape[:2] + (input_shape[2] * 2,)
    
class GarNet(keras.layers.Layer):
    """ """
    def __init__(self, n_aggregators, n_filters, n_propagate,
                 output_activation='tanh',
                 quantize_transforms=False,
                 simplified=False,
                 **kwargs):
        """ """
        
        super().__init__(**kwargs)
        
        self._simplified = simplified
        self._quantize_transforms = quantize_transforms
        self._output_activation=output_activation
        self._collapse = None # o un de 'mean', 'sum', 'max'
        self._mean_by_nvert = False
        
        self._setup_transforms(n_aggregators, n_filters, n_propagate)
        
    def _setup_transforms(self, n_aggregators, n_filters, n_propagate):
        """ """
        if self._quantize_transforms:
            self._input_feature_transform = NamedQDense(n_propagate, 
                                                        kernel_quantizer=ternary_1_05(),
                                                        bias_quantizer=ternary_1_05(),
                                                        name='FLR')
            self._output_feature_transform = NamedQDense(n_filters, 
                                                         activation=self._output_activation,
                                                         kernel_quantizer=ternary_1_05(),
                                                         name='Fout')
        else:
            self._input_feature_transform = NamedDense(n_propagate, name='FLR')
            self._output_feature_transform = NamedDense(n_filters, 
                                                        activation=self._output_activation, 
                                                        name='Fout')
        self._aggregator_distance = NamedDense(n_aggregators, name='S')
        self._sublayers = [self._input_feature_transform, self._aggregator_distance, self._output_feature_transform]

    def build(self, input_shape):
        """ """
        data_shape = input_shape
            
        self._build_transforms(data_shape)
        
        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)
        
    def _build_transforms(self, data_shape):
        """ """
        self._input_feature_transform.build(data_shape)
        if self._simplified:
            self._output_feature_transform.build(data_shape[:2] + (self._aggregator_distance.units * \
                                                                   self._input_feature_transform.units,))
        else:
            output_shape = data_shape[:2] + (data_shape[2] + 2 * self._aggregator_distance.units * \
                                            (self._aggregator_distance.units + self._input_feature_transform.units) + self._aggregator_distance.units,)
            
            self._output_feature_transform.build(output_shape)

    def call(self, x):
        """ """
        data, num_vertex, vertex_mask = self._unpack_input(x)
        
        output = self._garnet(data, num_vertex, vertex_mask,
                              self._input_feature_transform,
                              self._aggregator_distance,
                              self._output_feature_transform)
        
        output = self._collapse_output(output)
        
        return output
    
    def _unpack_input(self, x):
        """ """
        
        data = x
        data_shape = K.shape(data)
        
        B = data_shape[0]
        V = data_shape[1]
        
        ### Masks based on a specified number of vertices, works with ordered data
        
        #vertex_indices = K.tile(K.expand_dims(K.arange(0, V), axis=0), (B, 1))
        #vertex_mask = K.expand_dims(K.cast(K.less(vertex_indices, K.cast(num_vertex, 'int32')), 'float32'), axis=-1)
        
        ### Masks based on an energy cut off, compares last feature
        #energy_min = 150. #(GeV)
        
        energy_min = 0.5
        vertex_mask = K.cast(K.less(energy_min, data[..., 3:4]), 'float32')
        num_vertex = K.sum(vertex_mask)

        return data, num_vertex, vertex_mask

    
    def _garnet(self, data, num_verted, vertex_mask, in_transform, d_compute, out_transform):
        """ """
        features = in_transform(data) # (B, V, F)
        distance = d_compute(data) # (B, V, S)
        edge_weights = vertex_mask * K.exp(-K.square(distance)) # (B, V, S)
        
        if not self._simplified:
            features = K.concatenate([vertex_mask * features, edge_weights], axis=-1)
            
        if self._mean_by_nvert:
            def graph_mean(out, axis):
                s = K.sum(out, axis=axis)
                s = K.reshape(s, (-1, d_compute.units * in_transform.units) / num_vertex)
                s = K.reshape(s, (-1, d_compute.units, in_transform.units))
                return s
        else:
            graph_mean = K.mean
            
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)

        aggregated_mean = self._apply_edge_weights(features, edge_weights_trans, aggregation=graph_mean) # (B, S, F)

        
        if self._simplified:
            aggregated = aggregated_mean
        else:
            aggregated_max = self._apply_edge_weights(features, edge_weights_trans, aggregation=K.max)
            aggregated = K.concatenate([aggregated_max, aggregated_mean], axis=-1)

        updated_features = self._apply_edge_weights(aggregated, edge_weights) # (B, V, S*F)
        
        if not self._simplified:
            updated_features = K.concatenate([data, updated_features, edge_weights], axis=-1)
        updated_features = vertex_mask * out_transform(updated_features)
        
        return updated_features

    def _collapse_output(self, output):
        """ """
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
        """ """
        return self._get_output_shape(input_shape, self._output_feature_transform)
    
    def _get_output_shape(self, input_shape, output_transform):
        data_shape = input_shape
            
        if self._collapse is None:
            return data_shape[:2] + (out_transform.units,)
        else:
            return (data_shape[0], out_transform.units)
        
    def get_config(self):
        """ """
        config = super(GarNet, self).get_config()
        
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
        """ """
        config.update({
            'n_aggregators':self._aggregator_distance.units,
            'n_filters':self._output_feasture_transform.units,
            'n_propogate':self._input_feature_transform.units})
    
    @staticmethod
    def _apply_edge_weights(features, edge_weights, aggregation=None):
        """ """
        features = K.expand_dims(features, axis=1) # (B, 1, V, F)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, S, V, 1)
        out = edge_weights * features # (B, S, V, F)
        
        if aggregation:
            out = aggregation(out, axis=2)
        else:
            out = K.reshape(out, (-1, edge_weights.shape[1], features.shape[-1] * features.shape[-2]))
        return out
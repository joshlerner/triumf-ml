from contextlib import contextmanager
import sys, os
import shutil

gpu = input("GPU (n): ")
v = input("Vertices (n): ")
q = input("Quantized (q/c): ")
reset = input("Reset (y/n): ")
    
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

import hls4ml
import pickle

from PionReconstruction.util.Models import *
from PionReconstruction.util.Generators import *
from PionReconstruction.util.Plotting import *

model_path = '/fast_scratch_1/jlerner/data/models/'
out_path = '/fast_scratch_1/jlerner/GarNetHLS/'

if reset == 'y':
    shutil.rmtree(out_path + f'{q}_{v}')

if q == 'q':
	model = tf.keras.models.load_model(model_path + f'qGarNet_log_{v}')
elif q == 'c':
	model = tf.keras.models.load_model(model_path + f'GarNet_log_{v}')

config = {'Model': {'Precision': 'ap_fixed<22, 10>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
          'LayerName': {'data': {'Precision': {'result': 'ap_fixed<16, 6, AP_RND, AP_SAT>'}},
                        'vertex': {'Precision': {'result': 'ap_uint<16>'}},
                        'energy': {'Precision': {'result': 'ap_fixed<16, 6, AP_RND, AP_SAT>'}}},
          'Optimizers': ['eliminate_linear_activation']}

config['LayerType'] = {'InputLayer': {'ReuseFactor': 1, 'Trace': False},
                       'GarNetStack': {'ReuseFactor': 32, 'Trace': True}, 
                       'Dense': {'ReuseFactor': 1, 'Trace': True},
                       'Activation': {'ReuseFactor': 1, 'Trace': False}}

hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name=f'{q}_{v}',
                                                       output_dir=out_path + f'{q}_{v}',
                                                       part='xcu250-figd2104-2L-e')
hls_model.compile()

hls_model.build(csim=False)

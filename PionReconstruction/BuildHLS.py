from contextlib import contextmanager
import sys, os
import shutil

gpu = input("GPU: ")
    
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

import hls4ml
import pickle

from PionReconstruction.util.Models import *
from PionReconstruction.util.Generators import *
from PionReconstruction.util.Plotting import *


data_path = '/fast_scratch_1/atlas_images/v01-45/'

cell_geo_path = data_path + 'cell_geo.root'

out_path = '/fast_scratch_1/jlerner/data/'

model = tf.keras.models.load_model(out_path + f'models/qGarNet_log_128')

config = hls4ml.utils.config_from_keras_model(model, granularity='model')
config['Model']['Precision'] = 'ap_fixed<20, 10>'
config['LayerName'] = {'input_1': {'Precision': {'result': 'ap_fixed<16, 7, AP_RND, AP_SAT>'}},
                       'input_2': {'Precision': {'result': 'ap_uint<16>'}}}

config['LayerType'] = {'InputLayer': {'ReuseFactor': 1, 'Trace': False},
                       'GarNetStack': {'ReuseFactor': 1, 'Trace': True}, 
                       'Dense': {'ReuseFactor': 1, 'Trace': True},
                       'Activation': {'ReuseFactor': 1, 'Trace': False}}

hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name='GarNetHLS',
                                                       output_dir='/fast_scratch_1/jlerner/GarNetHLS/', 
                                                       part='xcu250-figd2104-2L-e')

hls_model.compile()

hls_model.build(csim=False, vsynth=True)

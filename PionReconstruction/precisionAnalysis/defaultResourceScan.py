from contextlib import contextmanager
import sys, os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Save HLS GarNet Model Accuracy ')
parser.add_argument('name', type=str, help='the name of the scan')
parser.add_argument('vmax', type=int, help='the number of cells in the model input')
parser.add_argument('precision', type=str, help='default precision (i.e. "<16, 6>")')
parser.add_argument('gpu', type=int, help='the gpu for analysis [0-7]')

args=parser.parse_args()
compact = "".join([ele for ele in args.precision if ele.isdigit()])
temp_dir = f'/fast_scratch_1/jlerner/tmp/{args.name}/{compact}'

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with suppress_stdout():
    import hls4ml

    from PionReconstruction.util.Models import *

    data_path = '/fast_scratch_1/atlas_images/v01-45/'

    cell_geo_path = data_path + 'cell_geo.root'

    out_path = '/fast_scratch_1/jlerner/data/'


    model = tf.keras.models.load_model(out_path + f'models/GarNet_log_{args.vmax}')
    qmodel = tf.keras.models.load_model(out_path + f'models/qGarNet_log_{args.vmax}')

    config = {'Model': {'Precision': f'ap_fixed{args.precision}', 'ReuseFactor': 1, 'Strategy': 'Latency'},
              
              'LayerName': {'data': {'Precision': {'result': 'ap_fixed<16, 6, AP_RND, AP_SAT>'}},
                            'vertex': {'Precision': {'result': 'ap_uint<10>'}},
                            'energy': {'Precision': {'result': 'ap_fixed<16, 6>'}}},
              'Optimizers': ['eliminate_linear_activation']}

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name=f'GarNetHLS',
                                                           output_dir=f'{temp_dir}/continuous/{args.vmax}', part='xcu250-figd2104-2L-e')
    hls_model.compile()
    
    qhls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name=f'GarNetHLS',
                                                            output_dir=f'{temp_dir}/quantized/{args.vmax}', part='xcu250-figd2104-2L-e')
    qhls_model.compile()

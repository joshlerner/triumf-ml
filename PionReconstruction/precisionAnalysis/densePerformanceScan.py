from contextlib import contextmanager
import sys, os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Save HLS GarNet Model Accuracy ')
parser.add_argument('name', type=str, help='the type of scan (i.e. "defaultFraction")')
parser.add_argument('vmax', type=int, help='the number of cells in the model input')
parser.add_argument('precision', type=str, help='dense precision (i.e. "<16, 6>")')
parser.add_argument('gpu', type=int, help='the gpu for analysis [0-7]')

args=parser.parse_args()

temp_dir = f'/fast_scratch_1/jlerner/tmp/{args.name}'
precisionFile = f'{os.getenv("HOME")}/projects/PionReconstruction/data/{args.name}.npz'

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
    import pickle

    from PionReconstruction.util.Models import *
    from PionReconstruction.util.Generators import *


    data_path = '/fast_scratch_1/atlas_images/v01-45/'

    cell_geo_path = data_path + 'cell_geo.root'

    out_path = '/fast_scratch_1/jlerner/data/'

    test_file_list = ([[data_path + f'pi0/user.angerami.24559740.OutputStream._000{i:03d}.root', 1] for i in range(232, 264)],
                      [[data_path + f'pipm/user.angerami.24559744.OutputStream._000{i:03d}.root', 0] for i in range(232, 264)])

    def rms(pred, target):
        mask = target > 0
        pred = pred[mask]
        target = target[mask]
        return np.sqrt(np.sum(np.square(pred - target)))

    model = tf.keras.models.load_model(out_path + f'models/GarNet_log_{args.vmax}')
    qmodel = tf.keras.models.load_model(out_path + f'models/qGarNet_log_{args.vmax}')

    test_generator = garnetDataGenerator(test_file_list,
                                         cell_geo_path,
                                         batch_size=20000,
                                         normalizer=('log', None),
                                         name=f'garnet_log_{args.vmax}',
                                         labeled=True,
                                         preprocess=False,
                                         output_dir=out_path + 'test/')

    x, y = next(test_generator.generator())

    keras_scaled_pred = np.exp(model.predict(x)[-1]*10).reshape(-1,)
    qkeras_scaled_pred = np.exp(qmodel.predict(x)[-1]*10).reshape(-1,)
    scaled_target = np.exp(y['regression']*10).reshape(-1,)

    Norm = rms(x[2], scaled_target)

    config = {'Model': {'Precision': 'ap_fixed<24, 12>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
              'LayerName': {'data': {'Precision': {'result': 'ap_fixed<16, 6, AP_RND, AP_SAT>'}},
                            'vertex': {'Precision': {'result': 'ap_uint<10>'}},
                            'energy': {'Precision': {'result': 'ap_fixed<16, 6>'}}},
              'LayerType': {'Dense': {'Precision': {'weight': f'ap_fixed{args.precision}'}}},
              'Optimizers': ['eliminate_linear_activation']}

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name='GarNetHLS',
                                                           output_dir=temp_dir, part='xcu280-fsvh2892-2L-e')
    hls_model.compile()
    data = np.load(precisionFile, allow_pickle=True)
    data = dict(data)
    hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)
    s = np.nan_to_num(Norm/rms(hls_scaled_pred, scaled_target), nan=0.0, neginf=0.0, posinf=0.0)
    data[args.name][()][args.precision]['Continuous'][f'{args.vmax}'] = s
    np.savez(precisionFile, **data)
    
    hls_model = hls4ml.converters.convert_from_keras_model(qmodel, hls_config=config, project_name='GarNetHLS',
                                                           output_dir=temp_dir, part='xcu280-fsvh2892-2L-e')
    hls_model.compile()
    data = np.load(precisionFile, allow_pickle=True)
    data = dict(data)
    hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)
    s = np.nan_to_num(Norm/rms(hls_scaled_pred, scaled_target), nan=0.0, neginf=0.0, posinf=0.0)
    data[args.name][()][args.precision]['Quantized'][f'{args.vmax}'] = s
    np.savez(precisionFile, **data)

data = np.load(precisionFile, allow_pickle=True)
data = dict(data)
print(f'{args.precision} - {args.vmax}:')
print(f"\t Continuous: {data[args.name][()][args.precision]['Continuous'][f'{args.vmax}']} \
        \t Quantized: {data[args.name][()][args.precision]['Quantized'][f'{args.vmax}']}")

from contextlib import contextmanager
import sys, os
import shutil

gpu = input("GPU: ")
temp_n = input("temp{n}: ")

parent_dir = '/home/joshualerner/projects/PionReconstruction/'
temp_dir = parent_dir + f'data/tmp{temp_n}/'
sys.path.insert(0, parent_dir)
try:
    os.mkdir(temp_dir)
except FileExistsError:
    pass

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
    
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import hls4ml
import pickle

from util.Models import *
from util.Generators import *
import matplotlib.pyplot as plt

data_path = '/fast_scratch_1/atlas_images/v01-45/'

cell_geo_path = data_path + 'cell_geo.root'

out_path = '/fast_scratch_1/jlerner/data/'

test_file_list = ([[data_path + f'pi0/user.angerami.24559740.OutputStream._000{i:03d}.root', 1] for i in range(232, 264)],
                  [[data_path + f'pipm/user.angerami.24559744.OutputStream._000{i:03d}.root', 0] for i in range(232, 264)])

def rms(pred, target):
    mask = target <= 1000
    pred = pred[mask]
    target = target[mask]
    return np.sqrt(np.sum(np.square(pred - target)))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot()

colors=['#733635ff', '#355373ff', '#357355ff']

integ = 10
precisions = [f'<{integ + frac}, {integ}>' for frac in range(6, 23)]
labels = []

for i, v in enumerate([32, 64, 128]):
    model = tf.keras.models.load_model(out_path + f'models/GarNet_log_{v}')
    qmodel = tf.keras.models.load_model(out_path + f'models/qGarNet_log_{v}')
    
    test_generator = garnetDataGenerator(test_file_list,
                                         cell_geo_path,
                                         batch_size=20000,
                                         normalizer=('log', None),
                                         name=f'garnet_log_{v}',
                                         labeled=True,
                                         preprocess=False,
                                         output_dir=out_path + 'test/')

    x, y = next(test_generator.generator())

    keras_scaled_pred = np.exp(model.predict(x)[-1]*10).reshape(-1,)
    qkeras_scaled_pred = np.exp(qmodel.predict(x)[-1]*10).reshape(-1,)
    scaled_target = np.exp(y['regression']*10).reshape(-1,)

    Norm = rms(x[2], scaled_target)
    
    cScan = []
    qScan = []
    
    print(f'\nDefault Fractional Bit Scan of {v} Vertex Model: ')
    
    for precision in precisions:
        config = {'Model': {'Precision': f'ap_fixed{precision}', 'ReuseFactor': 1, 'Strategy': 'Latency'},
                  'LayerName': {'input_1': {'Precision': {'result': 'ap_fixed<16, 7, AP_RND, AP_SAT>'}},
                                'input_2': {'Precision': {'result': 'ap_uint<16>'}}}}
        
        with suppress_stdout():
            hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name='GarNetHLS',
                                                                   output_dir=temp_dir, part='xcu250-figd2104-2L-e')
            hls_model.compile()
            
            hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)

            s = np.nan_to_num(Norm/rms(hls_scaled_pred, scaled_target), nan=0.0, neginf=0.0, posinf=0.0)
            cScan.append(s)

            hls_model = hls4ml.converters.convert_from_keras_model(qmodel, hls_config=config, project_name='GarNetHLS',
                                                                   output_dir=temp_dir, part='xcu250-figd2104-2L-e')
            hls_model.compile()
            
            hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)

            s = np.nan_to_num(Norm/rms(hls_scaled_pred, scaled_target), nan=0.0, neginf=0.0, posinf=0.0)
            qScan.append(s)
    
        print(f'\n{precision}\n\t Continuous: {cScan[-1]}\t Quantized: {qScan[-1]}')
    color = colors[i]
    ax.plot(precisions, cScan, color=color, label=f'{v} Cell Continuous')
    ax.plot(precisions, qScan, '--', color=color, label=f'{v} Cell Quantized')
    
ax.set_title('Scanning Default Fraction Bits')
ax.set_xlabel('Fixed Point Precision')
ax.set_ylabel('Regression Improvement Factor')
ax.tick_params(axis='x', labelrotation = 45)
ax.legend(loc='best')
fig.savefig(parent_dir + 'data/figures/ScanFraction')

shutil.rmtree(temp_dir)
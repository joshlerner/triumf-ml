from contextlib import contextmanager
import sys, os
import shutil

parent_dir = '/home/joshualerner/start_tf/PionReconstruction/'
temp_dir = parent_dir + 'data/temp3/'
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
    
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import hls4ml
import pickle

from util.Models import *
from util.Generators import *
import matplotlib.pyplot as plt

data_path = '/fast_scratch_1/atlas_images/v01-45/'

cell_geo_path = data_path + 'cell_geo.root'

out_path = '/fast_scratch_1/jlerner/data/'

model = tf.keras.models.load_model(out_path + 'models/GarNet_log')
qmodel = tf.keras.models.load_model(out_path + 'models/qGarNet_log')

test_file_list = ([[data_path + f'pi0/user.angerami.24559740.OutputStream._000{i:03d}.root', 1] 
                   for i in range(232, 264)],
                  [[data_path + f'pipm/user.angerami.24559744.OutputStream._000{i:03d}.root', 0] 
                   for i in range(232, 264)])

test_generator = garnetDataGenerator(test_file_list,
                                     cell_geo_path,
                                     batch_size=20000,
                                     normalizer=('log', None),
                                     name='garnet_log',
                                     labeled=True,
                                     preprocess=False,
                                     output_dir=out_path + 'test/')

x, y = next(test_generator.generator())

keras_scaled_pred = np.exp(model.predict(x)[-1]*10).reshape(-1,)
qkeras_scaled_pred = np.exp(qmodel.predict(x)[-1]*10).reshape(-1,)
scaled_target = np.exp(y['regression']*10).reshape(-1,)

def rms(pred, target):
    mask = target <= 100
    pred = pred[mask]
    target = target[mask]
    return np.sqrt(np.sum(np.square(pred - target)))

cNorm = rms(keras_scaled_pred, scaled_target)
qNorm = rms(qkeras_scaled_pred, scaled_target)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot()

cScan = []
qScan = []
labels = []

print('\nInput Fraction Bit Scan:')

for frac in range(5, 20):
    integ = 5
    bitwidth = frac + integ
    
    labels.append(f'<{bitwidth}, {integ}>')
    print(labels[-1])
    
    config = {'Model': {'Precision': 'ap_fixed<18, 8>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
              'LayerName': {'input_1': {'Precision': {'result': f'ap_fixed<{bitwidth}, {integ}, AP_RND, AP_SAT>'}},
                            'input_2': {'Precision': {'result': 'ap_uint<10>'}}}}
    with suppress_stdout():
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name='GarNetHLS',
                                                               output_dir=temp_dir,
                                                               part='xcku115-flvb2104-2-i')
        hls_model.compile()
    
    hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)
    s = np.nan_to_num(rms(hls_scaled_pred, scaled_target)/cNorm, nan=10.0)
    if s > 10:
        s = 10
    print(f'\t Continuous: {s}')
    cScan.append(s)
    del hls_model, hls_scaled_pred
    
    with suppress_stdout():
        hls_model = hls4ml.converters.convert_from_keras_model(qmodel, hls_config=config, project_name='GarNetHLS',
                                                               output_dir=temp_dir,
                                                               part='xcku115-flvb2104-2-i')
        hls_model.compile()

    hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)
    s = np.nan_to_num(rms(hls_scaled_pred, scaled_target)/qNorm, nan=10.0)
    if s > 10:
        s = 10
    print(f'\t Quantized: {s}')
    qScan.append(s)
    del hls_model, hls_scaled_pred

ax.plot(labels, cScan)
ax.plot(labels, qScan)

ax.set_title('Scanning Input Fraction Bits')
ax.set_xlabel('Fixed Point Precision')
ax.set_ylabel('HLS RMS / Keras RMS')
ax.tick_params(axis='x', labelrotation = 45)
ax.legend(labels=['Continuous','Quantized'])
fig.savefig(parent_dir + 'data/figures/ScanInputFraction')

shutil.rmtree(temp_dir)

del fig, ax, keras_scaled_pred, scaled_target, model, test_generator, cScan, qScan
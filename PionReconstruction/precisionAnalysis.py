import os
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

test_file_list = ([[data_path + f'pi0/user.angerami.24559740.OutputStream._000{i:03d}.root', 1] for i in range(232, 264)],
                  [[data_path + f'pipm/user.angerami.24559744.OutputStream._000{i:03d}.root', 0] for i in range(232, 264)])

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
scaled_target = np.exp(y['regression']*10).reshape(-1,)

def rms(pred, target):
    mask = target <= 100
    pred = pred[mask]
    target = target[mask]
    return np.sqrt(np.sum(np.square(pred - target)))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()

scanning = []
for frac in range(6, 17):
    config = {'Model': {'Precision': f'ap_fixed<{frac + 8}, 8>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
              'LayerName': {'input_1': {'Precision': {'result': 'ap_fixed<14, 5, AP_RND, AP_SAT>'}},
                            'input_2': {'Precision': {'result': 'ap_uint<10>'}}}}
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, project_name='GarNetHLS',
                                                           output_dir='/home/joshualerner/start_tf/PionReconstruction/data/temp/',
                                                           part='xcku115-flvb2104-2-i')
    hls_model.compile()
    hls_scaled_pred = np.exp(hls_model.predict(x)[-1]*10).reshape(-1,)

    scanning.append(rms(hls_scaled_pred, scaled_target)/rms(keras_scaled_pred, scaled_target))
    del hls_model, hls_scaled_pred

ax.plot([f'<{frac + 8}, 8>' for frac in range(6, 17)], scanning)

ax.set_title('Scanning Fractional Bits')
ax.set_xlabel('Fractional Bits')
ax.set_ylabel('HLS RMS / Keras RMS')
ax.set_yscale('log')
ax.legend(labels=labels)
fig.savefig('/home/joshualerner/start_tf/PionReconstruction/data/figures/ScanFraction')

del fig, ax, keras_scaled_pred, scaled_target, model, test_generator
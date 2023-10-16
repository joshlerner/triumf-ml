import matplotlib.pyplot as plt
import sys, os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Plot Precision Scan')
parser.add_argument('name', type=str, help='the precision layer that is scanned (i.e. "defaultFractionPerformance")')
args = parser.parse_args()

plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

#colors=['#733635ff', '#355373ff', '#357355ff']
colors = ['green', 'purple', 'darkorange']

precisionFile = f'{os.getenv("HOME")}/projects/PionReconstruction/data/{args.name}.npz'

data = np.load(precisionFile, allow_pickle=True)
data = dict(data)[args.name][()]

for i, vmax in enumerate(['32', '64']):
    color = colors[i]
    ax.plot(list(data.keys()), [float('nan') if d['Continuous'][vmax] == 0 
                                else d['Continuous'][vmax] for d in data.values()], color=color, label=f'{vmax} Cell Continuous')
    ax.plot(list(data.keys()), [float('nan') if d['Quantized'][vmax] == 0 
                                else d['Quantized'][vmax] for d in data.values()], '--', color=color, label=f'{vmax} Cell Quantized')
    
if 'Performance' in args.name:
    ylabel = 'Regression Improvement over Baseline'
elif 'Latency' in args.name:
    ylabel = 'Latency (µs)'
elif 'DSP' in args.name:
    ylabel = 'DSP Units'
elif 'LUT' in args.name:
    ylabel = 'LUT Units'
elif 'FF' in args.name:
    ylabel = 'FF Units'
    
    
    
ax.set_title(f'{args.name} Scan')
ax.set_xlabel('Fixed Point Precision')
ax.set_ylabel(ylabel)
ax.tick_params(axis='x', labelrotation = 45)
ax.legend(loc='best')
plt.tight_layout()
fig.savefig(f'/home/joshualerner/projects/PionReconstruction/data/figures/{args.name}')

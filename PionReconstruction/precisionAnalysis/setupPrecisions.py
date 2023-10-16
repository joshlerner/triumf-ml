import sys, os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Initialize Dictionary of Precisions for Scan.')
parser.add_argument('--start', type=int, default=4, dest='start', help='start of bit scan range (default: 4)')
parser.add_argument('--end', type=int, default=21, dest='end', help='end of bit scan range (default: 21)')
parser.add_argument('--width', type=int, default=16, dest='width', help='starting bit width (default: 16)')
parser.add_argument('name', type=str, help='the type of scan (i.e. defaultFractionPerformance)')

args=parser.parse_args()

precisionFile = f'{os.getenv("HOME")}/projects/PionReconstruction/data/{args.name}.npz'
try: 
    data = np.load(precisionFile, allow_pickle=True)
    data = dict(data)
except:
    data = {}
    
name = args.name.lower()

if 'fraction' in name:
    def precision(i):
        return f'<{i + args.width - args.start}, {args.width - args.start}>'
elif 'integer' in name:
    def precision(i):
        return f'<{i + args.width - args.start}, {i}>'

data[args.name] = {precision(i): {'Continuous': {'32': 0, '64': 0, '128': 0}, 'Quantized': {'32': 0, '64': 0, '128': 0}}
                   for i in range(args.start, args.end)}

np.savez(precisionFile, **data)

print("\n".join(k for k in data[args.name].keys()))


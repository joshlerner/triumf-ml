import re
import sys, os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Save HLS GarNet Model Accuracy ')
parser.add_argument('name', type=str, help='the name of the scan')
parser.add_argument('resource', type=str, help='the name of the resource')
parser.add_argument('vmax', type=int, help='the number of cells in the model input')
parser.add_argument('precision', type=str, help='default precision (i.e. "<16, 6>")')

args=parser.parse_args()
compact = "".join([ele for ele in args.precision if ele.isdigit()])
precisionFile = f'{os.getenv("HOME")}/projects/PionReconstruction/data/{args.resource}.npz'

def getLatency(file): 
    stop_idx = 100
    with open(file) as f:
        for idx, line in enumerate(f.readlines()):
            if f'+ Latency' in line:
                stop_idx = idx + 6
            if idx == stop_idx:
                return(int(re.findall(r'\d+', line)[0])*5/1000)
        return 0

def getDSP(synth_file): 
    stop_idx = 100
    with open(synth_file) as f:
        for idx, line in enumerate(f.readlines()):
            if 'Utilization Estimates' in line:
                stop_idx = idx + 14
            if idx == stop_idx:
                return(int(re.findall(r'\d+', line)[1]))
        return 0
    
def getFF(synth_file): 
    stop_idx = 100
    with open(synth_file) as f:
        for idx, line in enumerate(f.readlines()):
            if 'Utilization Estimates' in line:
                stop_idx = idx + 14
            if idx == stop_idx:
                return(int(re.findall(r'\d+', line)[2]))
        return 0
    
def getLUT(synth_file): 
    stop_idx = 100
    with open(synth_file) as f:
        for idx, line in enumerate(f.readlines()):
            if 'Utilization Estimates' in line:
                stop_idx = idx + 14
            if idx == stop_idx:
                return(int(re.findall(r'\d+', line)[3]))
        return 0

if 'Latency' in args.resource:
    def parseFunction(f):
        return getLatency(f)
elif 'DSP' in args.resource:
    def parseFunction(f):
        return getDSP(f)
elif 'FF' in args.resource:
    def parseFunction(f):
        return getFF(f)
elif 'LUT' in args.resource:
    def parseFunction(f):
        return getLUT(f)

data = np.load(precisionFile, allow_pickle=True)
data = dict(data)

try:
    synth_file = f'/fast_scratch_1/jlerner/tmp/{args.name}/{compact}/continuous/{args.vmax}/GarNetHLS_prj/solution1/syn/report/GarNetHLS_csynth.rpt'
    data[args.resource][()][args.precision]['Continuous'][f'{args.vmax}'] = parseFunction(synth_file)
except:
    print(f'No report found in {synth_file}')
try:
    synth_file = f'/fast_scratch_1/jlerner/tmp/{args.name}/{compact}/quantized/{args.vmax}/GarNetHLS_prj/solution1/syn/report/GarNetHLS_csynth.rpt'
    data[args.resource][()][args.precision]['Quantized'][f'{args.vmax}'] = parseFunction(synth_file)
except:
    print(f'No report found in {synth_file}')

np.savez(precisionFile, **data)

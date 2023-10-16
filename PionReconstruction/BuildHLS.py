"""
This script assists in running the build function on hls models from the command line without having to load in the hls model object. The build process can take a long time, but will run in the background on the CPUs.

"""

from contextlib import contextmanager
import argparse
import time
import sys, os

parser = argparse.ArgumentParser(description='Build HLS Model and Produce Synthesis Report')
parser.add_argument('model', type=str, help='path to the model project directory')
parser.add_argument('-c', '--csimulation', dest='csim', action='store_true', help='run C simulation')
parser.add_argument('-s', '--synthesis', dest='synth', action='store_true', help='run C/RTL synthesis')
parser.add_argument('-r', '--co-simulation', dest='cosim', action='store_true', help='run C/RTL co-simulation')
parser.add_argument('-v', '--validation', dest='val', action='store_true', help='run C/RTL validation')
parser.add_argument('-e', '--export', dest='export', action='store_true', help='export IP')
parser.add_argument('-l', '--vivado-synthesis', dest='vsynth', action='store_true', help='run Vivado Synthesis')
parser.add_argument('-f', '--fifo-opt', dest='fifo', action='store_true', help='Optimize FIFO')
parser.add_argument('-a', '--all', dest='all', action='store_true', help='run C simulation, C/RTL synthesis, C/RTL co-simulation, and Vivado synthesis')
parser.add_argument('--reset', dest='reset', action='store_true', help='remove any previous builds')


args=parser.parse_args()

if args.all:
    args.csim = True
    args.synth = True
    args.cosim = True
    args.vsynth = True
    
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

curr_dir = os.getcwd()
start = time.time()
os.chdir(args.model)

with suppress_stdout():
    os.system(f'vivado_hls -f build_prj.tcl "reset={args.reset} csim={args.csim} synth={args.synth} cosim={args.cosim} validation={args.val} export={args.export} vsynth={args.vsynth} fifo_opt={args.fifo}"')

os.chdir(curr_dir)
print(f'{args.model} finised in {time.time() - start}')
    


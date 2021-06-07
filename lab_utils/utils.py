
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info("Started...")

# cli
import argparse
class BooleanAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(BooleanAction, self).__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, False if option_string.startswith('--no') else True)

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--verbose', type=int, default=0, help='Verbose level: 0*')
ap.add_argument('--debug', '--no-debug', dest='debug', default=False, action=BooleanAction, help='Debug mode: F*')
ap.add_argument('--plot', '--no-plot', dest='plot', default=False, action=BooleanAction, help='Plot: F*')
ap.add_argument('--log', type=int, default=1, help='Tensorflow log level: 1*') # 1: suppress info, 2: warning, 3: error, 0: do not suppress
ap.add_argument('--test', type=int, default=0, help='test case no: 0*') 
ap.add_argument('--task', type=int, default=0, help='task no: 0*') 
ap.add_argument('--step', type=int, default=0, help='step no: 0*') 
args, extra_args = ap.parse_known_args()
logger.info(args)
# logger.info(extra_args)

if args.verbose:
    logger.setLevel(logging.DEBUG)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.log)

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import matplotlib as mp
if args.plot:
    mp.use('TkAgg')
import matplotlib.pyplot as plt

def debug():
    pass

def toc(fp):
    import re
    pattern = r'^### Step #[\d\s\w:-]+'
    matching_lines = [line for line in open(fp) if re.match(pattern, line)]
    print("\n#################################################")
    print(*matching_lines, sep="")


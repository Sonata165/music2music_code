'''
Tokenize and detokenize MIDI's to make the format consistant
For Composer's Assistant v2's inference
'''

import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(dirof(os.path.abspath(__file__)))))

from utils_common.utils import *
from remi_z import MultiTrack
from tqdm import tqdm

def main():
    observe_ts_dist_prob()

def procedures():
    observe_ts_dist_prob()


def observe_ts_dist_prob():
    ts_dist_fp = '/home/longshen/work/MuseCoco/musecoco/dataset_preparation/slakh/statistics/time_signature_dist/time_signature_dist.yaml'
    ts_dist = read_yaml(ts_dist_fp)

    # Convert values to percentage
    tot = sum(ts_dist.values())
    ts_dist = {k: v/tot for k, v in ts_dist.items()}

    save_dir = '/home/longshen/work/MuseCoco/musecoco/dataset_preparation/slakh/statistics/time_signature_dist'
    save_fp = jpath(save_dir, 'time_signature_dist_prob.json')
    save_yaml(ts_dist, save_fp)



if __name__ == '__main__':
    main()
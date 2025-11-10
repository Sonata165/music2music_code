import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from utils_common.utils import *
from utils_midi import remi_utils
from tqdm import tqdm



def main():
    '''
    Observe the piano distribution (i-0 and i-2)
    '''
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr'
    splits = ['valid', 'test', 'train']
    for split in splits:
        cnt = 0
        n_inst_dist = {}
        
        split_fp = jpath(data_dir, '{}.txt'.format(split))
        with open(split_fp) as f:
            data = f.readlines()
        for sample in data:
            has_i0 = False
            has_i2 = False
            sample = sample.strip().split()
            _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)
            insts = remi_utils.from_remi_get_insts(tgt_seq)
            n_inst = len(insts)

            update_dic_cnt(n_inst_dist, n_inst)

        # Convert value of drum_bar_cnt to percentage
        total = sum(n_inst_dist.values())
        for k in n_inst_dist:
            n_inst_dist[k] = '{}%'.format(round(n_inst_dist[k] / total * 100, 2)) 
        print(n_inst_dist)

if __name__ == '__main__':
    main()
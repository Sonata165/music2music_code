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
    Observe the drum distribution in the 8-bar dataset
    '''
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_8bar_lp4hop2_knorm_iquant_nost_ts44'
    splits = ['valid', 'test', 'train']
    for split in splits:
        cnt = 0
        drum_bar_cnt = {}
        split_fp = jpath(data_dir, '{}.txt'.format(split))
        with open(split_fp) as f:
            data = f.readlines()
        for sample in data:
            sample = sample.strip().split()
            _, tgt_seq = remi_utils.from_remi_eight_bar_split_hist_tgt_seq(sample)
            if 'i-128' in tgt_seq:
                cnt += 1
            drum_bar = tgt_seq.count('i-128')
            update_dic_cnt(drum_bar_cnt, drum_bar)
            # drum_seq = remi_utils.from_remi_get_drum_pos_and_pitch_seq_by_track(tgt_seq)
        print('{} split drum ratio: {}'.format(split, cnt / len(data)))
        
        # Convert value of drum_bar_cnt to percentage
        total = sum(drum_bar_cnt.values())
        for k in drum_bar_cnt:
            drum_bar_cnt[k] = '{}%'.format(round(drum_bar_cnt[k] / total * 100, 2)) 
        print(drum_bar_cnt)

if __name__ == '__main__':
    main()
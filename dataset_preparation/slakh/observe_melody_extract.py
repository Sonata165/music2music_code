import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from utils_common.utils import *
from utils_midi import remi_utils
from tqdm import tqdm


def main():
    count_non_empty_melody()


def count_non_empty_melody():
    '''
    Observe the proportion of samples where melody can be successfully extracted
    '''
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr'
    splits = ['valid', 'test', 'train']
    
    for split in splits:
        cnt = 0
        split_fp = jpath(data_dir, '{}.txt'.format(split))
        with open(split_fp) as f:
            data = f.readlines()
        lines = [line.strip().split() for line in data]
        for line in tqdm(lines):
            _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(line)
            melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
                tgt_seq,
                monophonic_only=True,
                top_note=False
            )
            if len(melody_seq) == 0:
                cnt += 1
        print('{} split mel non empty ratio: {}'.format(split, 1 - cnt / len(lines)))
        


if __name__ == '__main__':
    main()
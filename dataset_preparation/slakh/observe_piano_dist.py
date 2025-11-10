import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from utils_common.utils import *
from utils_midi import remi_utils
from tqdm import tqdm



def main():
    piano_range()

def procedures():
    ap_ep()
    monophonic()
    piano_range()

def ap_ep():
    '''
    Observe the piano distribution (i-0 and i-2)
    '''
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr'
    splits = ['valid', 'test', 'train']
    for split in splits:
        cnt = 0
        piano_dist_cnt = {}
        
        split_fp = jpath(data_dir, '{}.txt'.format(split))
        with open(split_fp) as f:
            data = f.readlines()
        for sample in data:
            has_i0 = False
            has_i2 = False
            sample = sample.strip().split()
            _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)
            if 'i-0' in tgt_seq:
                has_i0 = True
            if 'i-2' in tgt_seq:
                has_i2 = True

            if has_i0 and has_i2:
                update_dic_cnt(piano_dist_cnt, 'both')
            elif has_i0:
                update_dic_cnt(piano_dist_cnt, 'i-0_only')
            elif has_i2:
                update_dic_cnt(piano_dist_cnt, 'i-2_only')
            else:
                update_dic_cnt(piano_dist_cnt, 'none')

        # Convert value of drum_bar_cnt to percentage
        total = sum(piano_dist_cnt.values())
        for k in piano_dist_cnt:
            piano_dist_cnt[k] = '{}%'.format(round(piano_dist_cnt[k] / total * 100, 2)) 
        print(piano_dist_cnt)


def monophonic():
    '''
    Observe the piano distribution (i-0 and i-2)
    '''
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr'
    splits = ['valid', 'test', 'train']
    for split in splits:
        cnt = 0
        piano_dist_cnt = {}
        
        split_fp = jpath(data_dir, '{}.txt'.format(split))
        with open(split_fp) as f:
            data = f.readlines()
        for sample in data:
            sample = sample.strip().split()
            _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)

            # Get piano opd
            ap_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
            ep_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
            t = {'i-0': ap_opd, 'i-2': ep_opd}
            piano_opd = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(t)

            pos_tok = [tok for tok in piano_opd if tok.startswith('o')]
            pitch_tok = [tok for tok in piano_opd if tok.startswith('p')]
            n_pos = len(pos_tok)
            n_pitch = len(pitch_tok)

            if n_pos == 0:
                update_dic_cnt(piano_dist_cnt, 'none')            
            elif n_pos == n_pitch:
                update_dic_cnt(piano_dist_cnt, 'monophonic')
            else:
                update_dic_cnt(piano_dist_cnt, 'polyphonic')

        # Convert value of drum_bar_cnt to percentage
        total = sum(piano_dist_cnt.values())
        for k in piano_dist_cnt:
            piano_dist_cnt[k] = '{}%'.format(round(piano_dist_cnt[k] / total * 100, 2)) 
        print(piano_dist_cnt)

def piano_range():
    '''
    Observe the piano distribution (i-0 and i-2)
    '''
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr'
    splits = ['valid', 'test', 'train']
    for split in splits:
        cnt = 0
        # piano_dist_cnt = {}
        piano_range_ratio = []
        
        split_fp = jpath(data_dir, '{}.txt'.format(split))
        with open(split_fp) as f:
            data = f.readlines()
        for sample in data:
            sample = sample.strip().split()
            _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)

            # Get piano opd
            ap_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
            ep_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
            t = {'i-0': ap_opd, 'i-2': ep_opd}
            piano_opd = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(t)

            # Get piano range
            piano_pitch = [int(tok.split('-')[1]) for tok in piano_opd if tok.startswith('p')]
            if len(piano_pitch) == 0:
                piano_range = 0
            else:
                piano_range = max(piano_pitch) - min(piano_pitch)

            # Get entire range
            entire_pitch = [int(tok.split('-')[1]) for tok in tgt_seq if tok.startswith('p')]
            if len(entire_pitch) == 0:
                ratio = -1
            else:
                entire_range = max(entire_pitch) - min(entire_pitch)
                if entire_range == 0:
                    ratio = -1
                else:
                    ratio = piano_range / entire_range
            piano_range_ratio.append(ratio)
            # update_dic_cnt(piano_range_ratio, ratio)

        # Plot histogram
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(piano_range_ratio, bins=100)
        plt.xlabel('Piano range ratio')
        plt.ylabel('Count')
        plt.title('Piano range ratio distribution')
        save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics/piano_range_ratio'
        save_fp = jpath(save_dir, 'piano_range_ratio_{}.png'.format(split))
        plt.savefig(save_fp)

        # Compute qunatile
        qs = [0, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1]
        q_vals = np.quantile(piano_range_ratio, qs)
        save_json_fp = jpath(save_dir, 'piano_range_ratio_{}.json'.format(split))
        save_json({q: v for q, v in zip(qs, q_vals)}, save_json_fp) 


        # # Convert value of piano_range_ratio to percentage
        # total = sum(piano_range_ratio.values())
        # for k in piano_range_ratio:
        #     piano_range_ratio[k] = '{}%'.format(round(piano_range_ratio[k] / total * 100, 2))
        # print(piano_range_ratio)


if __name__ == '__main__':
    main()
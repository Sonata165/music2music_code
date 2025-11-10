'''
Old version

Create a 8-bar segmented version for the slakh dataset
- Each sample contains 8-bars
- Between samples there are 6-bar overlaps (hop_length=2 bars)
- 4-bar empty padding to the beginning
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.abspath('../..'))

from src_hf.utils import *
from utils_midi import remi_utils
from tqdm import tqdm 


def main():
    procedures()


def procedures():
    '''
    # ----- Archived -----
    segment_8bar()
    segment_2bar()
    segment_4bar_pad0_hop2()
    segment_2bar_normalized()

    segment_2bar_keynormalized_instquantized_nos_not()
    segment_4bar_keynormalized_instquantized_nos_not()
    segment_2bar_keynormalized_instquantized_nos_not_44()
    segment_2bar_keynormalized_instquantized_nost_44_nodr()
    segment_8bar_lp4hop2_keynorm_instquant_nost_44()
    '''

    # For band arrangement
    segment_2bar_keynormalized_instquantized_nost_44_nodr_remi()

    # For piano reduction
    segment_2bar_keynormalized_instquantized_nost_44_nodr()

    # For drum arrangement
    segment_8bar_lp4hop2_keynorm_instquant_nost_44()


def segment_2bar_keynormalized_instquantized_nost_44_nodr_remi():
    '''Use remi+, not remi-z'''
    segment_slakh_remi_norm_quant_nost(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr',
        segment_n_bar=2,
        hop_n_bar=1,
        front_pad_bar=0,
        only_4_4=True,
        no_drum=True,
    )


def segment_2bar_keynormalized_instquantized_nost_44_nodr():
    segment_slakh_remi_norm_quant_nost(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44_nodr',
        segment_n_bar=2,
        hop_n_bar=1,
        front_pad_bar=0,
        only_4_4=True,
        no_drum=True,
    )


def segment_8bar_lp4hop2_keynorm_instquant_nost_44():
    segment_slakh_remi_norm_quant_nost(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_8bar_lp4hop2_knorm_iquant_nost_ts44',
        segment_n_bar=8,
        hop_n_bar=2,
        front_pad_bar=4,
        only_4_4=True
    )


def segment_2bar_keynormalized_instquantized_nos_not_44():
    segment_slakh_remi_norm_quant_nost(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost_ts44',
        segment_n_bar=2,
        hop_n_bar=1,
        front_pad_bar=0,
        only_4_4=True
    )


def segment_4bar_keynormalized_instquantized_nos_not():
    segment_slakh_remi_norm_quant_nost(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_4bar_lpad0_hop2_norm_quant_nost_ts44',
        segment_n_bar=4,
        hop_n_bar=2,
        front_pad_bar=0,
        only_4_4=True
    )


def segment_2bar_keynormalized_instquantized_nos_not():
    segment_slakh_remi_norm_quant_nost(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_norm_quant_nost',
        segment_n_bar=2,
        hop_n_bar=1,
        front_pad_bar=0
    )


def segment_2bar_normalized():
    segment_slakh_remi(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI_normalized',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1_normalized',
        segment_n_bar=2,
        hop_n_bar=1,
        front_pad_bar=0
    )


def segment_4bar_pad0_hop2():
    segment_slakh_remi(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_4bar_lpad0_hop2',
        segment_n_bar=4,
        hop_n_bar=2,
        front_pad_bar=0
    )

def segment_8bar():
    segment_slakh_remi(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_8bar',
        segment_n_bar=8,
        hop_n_bar=2,
        front_pad_bar=4
    )

def segment_2bar():
    segment_slakh_remi(
        data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI',
        out_dir= '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar',
        segment_n_bar=2,
        hop_n_bar=1,
        front_pad_bar=0
    )


def segment_slakh_remi(data_dir, out_dir, segment_n_bar=8, hop_n_bar=2, front_pad_bar=4):
    '''
    Generate the 8bar dataset for slakh2100
    '''
    dataset_dir = data_dir

    create_dir_if_not_exist(out_dir)
    split_dirnames = ['validation', 'test', 'train']
    for split_dirname in split_dirnames:
        split_data = []

        split = split_dirname if split_dirname != 'validation' else 'valid'
        split_dirpath = jpath(dataset_dir, split_dirname)
        split_out_fp = jpath(out_dir, split + '.txt')

        song_fns = ls(split_dirpath)
        for song_fn in song_fns:
            song_fp = jpath(split_dirpath, song_fn)

            with open(song_fp, 'r') as f:
                song_remi = f.read().strip()
            song_remi_seq = song_remi.split(' ')

            # Obtain bar positions
            remi_of_all_bars = []
            bar_indices = remi_utils.from_remi_get_bar_idx(song_remi_seq)
            for bar_id in bar_indices:
                bar_start_idx, bar_end_idx = bar_indices[bar_id]
                bar_seq = song_remi_seq[bar_start_idx:bar_end_idx]
                
                # Add 4 empty bars to the beginning
                if len(remi_of_all_bars) == 0:
                    # Generate a pseudo empty bar, with same time signature and tempo as the first bar
                    empty_bar = bar_seq[:2] + ['b-1']
                    for _ in range(front_pad_bar):
                        remi_of_all_bars.append(empty_bar)

                remi_of_all_bars.append(bar_seq)

            # Generate 8-bar segments
            for i in range(0, len(remi_of_all_bars) - segment_n_bar, hop_n_bar):
                segment = []
                for j in range(i, i +segment_n_bar):
                    segment.extend(remi_of_all_bars[j])
                split_data.append(segment)

        # Save the split data
        with open(split_out_fp, 'w') as f:
            for segment in split_data:
                f.write(' '.join(segment) + '\n')


def segment_slakh_remi_norm_quant_nost(data_dir, 
                                       out_dir, 
                                       segment_n_bar=8, 
                                       hop_n_bar=2, 
                                       front_pad_bar=4, 
                                       only_4_4=False,
                                       no_drum=False,
                                       ):
    '''
    Generate the 8bar dataset for slakh2100
    '''
    dataset_dir = data_dir

    create_dir_if_not_exist(out_dir)
    split_dirnames = ['validation', 'test', 'train']
    from utils_instrument.inst_map import InstMapUtil
    inst_util = InstMapUtil()
    for split_dirname in split_dirnames:
        split_data = []

        split = split_dirname if split_dirname != 'validation' else 'valid'
        split_dirpath = jpath(dataset_dir, split_dirname)
        split_out_fp = jpath(out_dir, split + '.txt')

        song_fns = ls(split_dirpath)
        for song_fn in tqdm(song_fns):
            song_fp = jpath(split_dirpath, song_fn)

            with open(song_fp, 'r') as f:
                song_remi = f.read().strip()
            song_remi_seq = song_remi.split(' ')

            if only_4_4:
                ts_token = song_remi_seq[0]
                if ts_token != 's-9':
                    continue

            # Obtain bar positions
            remi_of_all_bars = []
            bar_indices = remi_utils.from_remi_get_bar_idx(song_remi_seq)
            for bar_id in bar_indices:
                bar_start_idx, bar_end_idx = bar_indices[bar_id]
                bar_seq = song_remi_seq[bar_start_idx:bar_end_idx]

                # Instrument quantization
                opd_seq_of_track = remi_utils.from_remi_get_opd_seq_per_track(bar_seq, sort_by_avg_pitch=True)
                new_content = {}
                for inst in opd_seq_of_track:
                    inst_id = inst_util.slakh_quantize_inst_prog(int(inst.split('-')[1]))
                    if no_drum and inst_id == 128:
                        continue
                    if inst_id is not None:
                        new_content['i-{}'.format(inst_id)] = opd_seq_of_track[inst]
                new_bar_seq = []
                for inst in new_content:
                    new_bar_seq.append(inst)
                    new_bar_seq.extend(new_content[inst])
                new_bar_seq.append('b-1')
                bar_seq = new_bar_seq
                
                # Add 4 empty bars to the beginning
                if len(remi_of_all_bars) == 0:
                    # Generate a pseudo empty bar, with same time signature and tempo as the first bar
                    empty_bar = ['b-1']
                    for _ in range(front_pad_bar):
                        remi_of_all_bars.append(empty_bar)

                remi_of_all_bars.append(bar_seq)

            # Generate 8-bar segments
            for i in range(0, len(remi_of_all_bars) - segment_n_bar, hop_n_bar):
                segment = []
                for j in range(i, i +segment_n_bar):
                    segment.extend(remi_of_all_bars[j])
                split_data.append(segment)

        # Save the split data
        with open(split_out_fp, 'w') as f:
            for segment in split_data:
                f.write(' '.join(segment) + '\n')




if __name__ == '__main__':
    main()
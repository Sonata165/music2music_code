import os
import sys

sys.path.append('../..')

from utils_common.utils import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    get_tempo_distribution()


def procedures():
    count_token_per_sample()
    get_ts_distribution()
    get_tempo_distribution()


def get_tempo_distribution():
    '''
    Get the tempo distribution of the dataset
    '''
    tempo_dict = read_yaml('/home/longshen/work/musecoco/utils_midi/tempo_dict.yaml')
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI'
    ts = {}
    splits = ['validation', 'test', 'train']
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        song_fns = ls(split_dirpath)
        for song_fn in song_fns:
            song_fp = jpath(split_dirpath, song_fn)
            with open(song_fp, 'r') as f:
                song = f.read()
            song_seq = song.strip().split()
            for i, tok in enumerate(song_seq):
                if tok.startswith('t'):
                    # ts_val = '{}, ({})'.format(ts_dict[tok], tok)
                    tempo = '{}, ({})'.format(tempo_dict[tok], tok)
                    update_dic_cnt(ts, tempo)

    save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics/tempo_dist'
    create_dir_if_not_exist(save_dir)

    # Sort by value
    ts = dict(sorted(ts.items(), key=lambda item: int(item[0].split(',')[1].split(')')[0])))
    save_fp = jpath(save_dir, 'tempo_dist.yaml')
    save_yaml(ts, save_fp)

    # Draw a horizontal barchart
    fig, ax = plt.subplots()
    # Draw a horizontal barchart
    ax.bar(list(ts.keys()), list(ts.values()))
    ax.set_xlabel('Tempo')
    # Rotate x label 90 degree
    plt.xticks(rotation=90)
    ax.set_ylabel('Count')
    # Log-scale y axis
    # ax.set_xscale('log')
    ax.set_title('Tempo Distribution of Slakh2100')
    plt.tight_layout()
    fig_save_fp = jpath(save_dir, 'tempo_dist.png')
    plt.savefig(fig_save_fp)


def get_ts_distribution():
    '''
    Get the time signature distribution of the dataset
    '''
    ts_dict = read_yaml('/home/longshen/work/musecoco/utils_midi/ts_dict.yaml')
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI'
    ts = {}
    splits = ['validation', 'test', 'train']
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        song_fns = ls(split_dirpath)
        for song_fn in song_fns:
            song_fp = jpath(split_dirpath, song_fn)
            with open(song_fp, 'r') as f:
                song = f.read()
            song_seq = song.strip().split()
            for i, tok in enumerate(song_seq):
                if tok.startswith('s'):
                    ts_val = '{}, ({})'.format(ts_dict[tok], tok)
                    update_dic_cnt(ts, ts_val)

    save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics/time_signature_dist'
    create_dir_if_not_exist(save_dir)

    # sort by value
    ts = dict(sorted(ts.items(), key=lambda item: item[1], reverse=True))
    save_fp = jpath(save_dir, 'time_signature_dist.yaml')
    save_yaml(ts, save_fp)

    # Draw a horizontal barchart
    fig, ax = plt.subplots()
    # Draw a horizontal barchart
    ax.barh(list(ts.keys()), list(ts.values()))
    ax.set_xlabel('Time Signature')
    # Rotate x label 90 degree
    # plt.xticks(rotation=45)
    ax.set_ylabel('Count')
    # Log-scale y axis
    ax.set_xscale('log')
    ax.set_title('Time Signature Distribution of Slakh2100')
    plt.tight_layout()
    fig_save_fp = jpath(save_dir, 'time_signature_dist.png')
    plt.savefig(fig_save_fp)
    



def count_token_per_sample():
    data_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_2bar_lpad0_hop1'
    splits = ['train', 'valid', 'test']
    res = ''
    for split in splits:
        split_fp = os.path.join(data_dir, split) + '.txt'

        with open(split_fp, 'r') as f:
            data = f.readlines()
        data = [line.strip().split() for line in data]
        n_tokens = [len(tokens) for tokens in data]

        # Print quantiles computed by numpy
        qs = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        print(f'{split} quantiles:')
        res += f'{split} quantiles:\n'
        for q in qs:
            print(f'  {q*100}th: {np.quantile(n_tokens, q)}')
            res += f'  {q*100}th: {np.quantile(n_tokens, q)}\n'
        res += '\n'
    
    save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics'
    save_fn = 'n_tokens_2bar.txt'
    save_fp = os.path.join(save_dir, save_fn)
    with open(save_fp, 'w') as f:
        f.write(res)
    


if __name__ == '__main__':
    main()
import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(dirof(os.path.abspath(__file__)))))

from utils_common.utils import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # get_drum_coverage()
    get_drum_coverage_test_split()


def procedures():
    count_token_per_sample()
    get_ts_distribution()
    get_tempo_distribution()


def get_drum_coverage_test_split():
    '''
    Get the tempo distribution of the dataset
    '''
    data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json'
    data = read_json(data_fp)
    res_dir = '/home/longshen/work/MuseCoco/musecoco/dataset_preparation/slakh/statistics/drum_coverage'
    create_dir_if_not_exist(res_dir)
    res_fn = 'drum_coverage_test_split_44.json'
    res_fp = jpath(res_dir, res_fn)

    res = {}
    ts = {}
    splits = ['test']
    non_44_tracks = set()
    for split in splits:
        split_data = data[split]
        
        for bar_name in split_data:
            bar = split_data[bar_name]
            track_name, bar_idx = bar_name.split('-')

            if bar['meta']['time_signature'] != '(4, 4)':
                non_44_tracks.add(track_name)
                continue

            if track_name not in res:
                res[track_name] = []

            
            if bar['meta']['has_drum']:
                res[track_name].append(1)
            else:
                res[track_name].append(0)

    # Remove non-4/4 tracks
    for track_name in non_44_tracks:
        res.pop(track_name, None)

    # Calculate the drum coverage for each song
    cov = {}
    for track_name in res:
        cov[track_name] = np.mean(res[track_name])    


    # Sort by value
    cov = dict(sorted(cov.items(), key=lambda item: item[1], reverse=True))

    save_json(cov, res_fp)

    # # Draw a horizontal barchart
    # fig, ax = plt.subplots()
    # # Draw a horizontal barchart
    # ax.bar(list(ts.keys()), list(ts.values()))
    # ax.set_xlabel('Tempo')
    # # Rotate x label 90 degree
    # plt.xticks(rotation=90)
    # ax.set_ylabel('Count')
    # # Log-scale y axis
    # # ax.set_xscale('log')
    # ax.set_title('Tempo Distribution of Slakh2100')
    # plt.tight_layout()
    # fig_save_fp = jpath(save_dir, 'tempo_dist.png')
    # plt.savefig(fig_save_fp)


def get_drum_coverage():
    '''
    Get the tempo distribution of the dataset
    '''
    data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json'
    data = read_json(data_fp)
    res_dir = '/home/longshen/work/MuseCoco/musecoco/dataset_preparation/slakh/statistics'
    res_fn = 'drum_coverage.json'
    res_fp = jpath(res_dir, res_fn)

    res = {}
    ts = {}
    splits = ['validation', 'test', 'train']
    for split in splits:
        split_data = data[split]
        for bar_name in split_data:
            track_name, bar_idx = bar_name.split('-')
            if track_name not in res:
                res[track_name] = []

            bar = split_data[bar_name]
            if bar['meta']['has_drum']:
                res[track_name].append(1)
            else:
                res[track_name].append(0)

    # Calculate the drum coverage for each song
    cov = {}
    for track_name in res:
        cov[track_name] = np.mean(res[track_name])    


    # Sort by value
    cov = dict(sorted(cov.items(), key=lambda item: item[1], reverse=True))

    save_json(cov, res_fp)

    # # Draw a horizontal barchart
    # fig, ax = plt.subplots()
    # # Draw a horizontal barchart
    # ax.bar(list(ts.keys()), list(ts.values()))
    # ax.set_xlabel('Tempo')
    # # Rotate x label 90 degree
    # plt.xticks(rotation=90)
    # ax.set_ylabel('Count')
    # # Log-scale y axis
    # # ax.set_xscale('log')
    # ax.set_title('Tempo Distribution of Slakh2100')
    # plt.tight_layout()
    # fig_save_fp = jpath(save_dir, 'tempo_dist.png')
    # plt.savefig(fig_save_fp)


    


if __name__ == '__main__':
    main()
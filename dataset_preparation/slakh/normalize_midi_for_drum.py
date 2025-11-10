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
    normalize_midis_for_drum()

def procedures():
    pass


def normalize_midis_for_drum():
    # Get the track names to normalize
    track_name_fp = '/home/longshen/work/MuseCoco/musecoco/dataset_preparation/slakh/statistics/drum_coverage/drum_coverage_test_split_44.json'
    track_name_dict = read_json(track_name_fp)
    track_names = list(track_name_dict.keys())

    midi_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/original/test'
    out_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/test_normalized'
    for track_name in tqdm(track_names):
        in_midi_fp = jpath(midi_dir, track_name, 'all_src.mid')
        out_midi_fp = jpath(out_dir, f'{track_name}.mid')

        mt = MultiTrack.from_midi(in_midi_fp)
        mt = mt[:64] # Only take the first 64 bars

        mt.to_midi(out_midi_fp)


if __name__ == '__main__':
    main()
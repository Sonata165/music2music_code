import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from src.utils import get_dataset_loc, get_dataset_dir, ls, jpath
from .slakh_jingwei import get_piano_rolls_and_other_features_from_mix_midi


def _main():
    test_quantize()


def _procedures():
    pass


def test_quantize():
    h5_fp = get_dataset_loc()
    h5_data = h5py.File(h5_fp, 'r')
    data_dir = get_dataset_dir()
    for song_name in tqdm(h5_data):
        song_entry = h5_data[song_name]
        split_name = song_entry.attrs['split']
        track_dpath = jpath(data_dir, split_name, song_name)
        assert os.path.exists(track_dpath)
        mt3_out_fpath = jpath(track_dpath, 'detected_chords.txt')
        assert os.path.exists(mt3_out_fpath)

        chord_tuple_list = read_detected_chord(mt3_out_fpath)

        pos_of_16th_note = song_entry['pos_in_sec'][()]
        chord_seq = quantize_chord(
            detected_chords=chord_tuple_list,
            pos_16th=pos_of_16th_note
        )

        song_entry.create_dataset('chord_seq', data=chord_seq)

    h5_data.close()


def read_detected_chord(chord_fp):
    '''
    Read info from chord recognition result file
    :param chord_fp: chord recognition output path
    :return: a list of tuples, each tuple contains (1) float, onset (2) float, offset (3) str, chord type
    '''
    assert os.path.exists(chord_fp)
    with open(chord_fp) as f:
        results = f.readlines()
    ret = []
    for chord in results:
        onset, offset, chord_type = chord.strip().split('\t')
        onset = float(onset)
        offset = float(offset)
        ret.append((onset, offset, chord_type))
    return ret


def quantize_chord(detected_chords, pos_16th, num_beat_per_chord=1):
    '''
    Convert chord recognition results to a quantized chord sequence.
    For an entire song.
    Each element inside the list represent one beat.
    Each chord is represented by an integer within the range of [1, 26],
    following the chord definition of "submission_chord_list.txt"

    :param detected_chords: A list of tuples:
        [(onset1, offset1, chord1), (onset2, offset2, chord2), ...]
    :param pos_16th: time of all 16th note's position, in second
        NOTE: It represents the STARTING TIME of each position. Starting from 0.
    :return: a list of chord ids.
    '''
    chord_pos = pos_16th[0::4 * num_beat_per_chord]  # The starting time of each chord, in second
    num_chord_symbols = len(chord_pos)
    # ret = np.zeros(shape=num_chord_symbols, dtype='S16')  # Use byte string to support hdf5
    ret = np.full(shape=num_chord_symbols, fill_value='N', dtype='U16')  # Use unicode string
    chord_pos = np.append(chord_pos, chord_pos[1] - chord_pos[0] + chord_pos[-1])
    for onset, offset, chord_type in detected_chords:
        # Get the nearest
        # onset_beat_id = np.searchsorted(beat_pos, onset)
        onset_beat_id, _ = find_nearest(chord_pos, onset)
        offset_beat_id, _ = find_nearest(chord_pos, offset)
        # offset_beat_id = np.searchsorted(beat_pos, offset)
        ret[onset_beat_id:offset_beat_id] = chord_type
    return ret


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


if __name__ == '__main__':
    _main()

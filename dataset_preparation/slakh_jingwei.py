'''
Generate quantized notation from midi files.
'''

import os
import sys
import time

import numpy as np
import pretty_midi as pyd
from tqdm import tqdm
from scipy.interpolate import interp1d
import yaml

from utils_instrument.inst_map import InstMapUtil

sys.path.insert(1, os.path.join(sys.path[0], '../'))


def _main():
    test_get_proll_from_mix()


def quantize_slakh():
    '''
    Original code, from Jingwei, to generate quantized piano roll (song-level) for the entire Slakh dataset
    '''
    pos_per_beat = 4  # quantize each beat as 4 positions

    # Before running the code, get raw MIDI of Slakh2100 from https://zenodo.org/record/4599666.
    slakh_root = '/data1/longshen/Datasets/slakh2100_flac_redux'
    save_root = '/data1/longshen/Datasets/slakh2100_flac_redux/quantized'

    for split in ['train', 'validation', 'test']:
        # for split in ['test']: # for debugging
        split_path = os.path.join(slakh_root, split)
        split_save_path = os.path.join(save_root, split)
        if not os.path.exists(split_save_path):
            os.makedirs(split_save_path)
        print(f'processing {split} set ...')
        songs = os.listdir(split_path)
        songs.sort()
        for song in tqdm(songs):
            if song == '.DS_Store':
                continue
            break_flag = 0

            # Read a song with multiple tracks
            all_src_midi = pyd.PrettyMIDI(os.path.join(split_path, song, 'all_src.mid'))

            # Process only 2/4 and 4/4 songs
            for ts in all_src_midi.time_signature_changes:
                if ts.denominator == 4 and (ts.numerator == 2 or ts.numerator == 4):
                    continue
                else:
                    break_flag = 1
                    break
            if break_flag:
                continue

            # Read separate midi file for each track
            track_paths = os.path.join(split_path, song, 'MIDI')
            track_names = os.listdir(track_paths)
            track_midis = [pyd.PrettyMIDI(os.path.join(track_paths, track)) for track in track_names]

            # Read metadata file for the song
            metadata = yaml.safe_load(open(os.path.join(split_path, song, 'metadata.yaml'), 'r'))['stems']

            # Choose the longest midi file, obtain the beat positions (in second)
            if len(all_src_midi.get_beats()) >= max([len(midi.get_beats()) for midi in track_midis]):
                longest_track_midi = all_src_midi
            else:
                # raise Exception('Track midi longer than mix midi: {}'.format(track_paths))
                longest_track_midi_index = np.argmax([len(midi.get_beats()) for midi in track_midis])
                longest_track_midi = track_midis[longest_track_midi_index]
            # start_time = longest_track_midi.estimate_beat_start(candidates=10, tolerance=0.025)
            start_time = 0
            beats = longest_track_midi.get_beats(start_time=start_time)
            downbeats = longest_track_midi.get_downbeats(start_time=start_time)

            # Get all 16th note position
            beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))  # 补充上结束时间点

            # [TODO] 暂时丢掉弱起小节
            # # 给弱起添加一个多余的小节
            # bar_dur = downbeats[1]-downbeats[0]
            # init_downbeat = downbeats[0]-bar_dur
            # downbeats = np.insert(downbeats, 0, init_downbeat)
            # beat_dur = bar_dur / ts.denominator
            # additional_beats = np.arange(init_downbeat, beats[0], beat_dur)

            quantize = interp1d(np.arange(0, len(beats)) * pos_per_beat, beats, kind='linear')
            pos_16th = quantize(np.arange(0, (len(beats) - 1) * pos_per_beat))  # Get position of each

            track_name_list = []
            all_piano_rolls = []
            programs = []
            dynamic_mats = []
            onset_mats = []
            offset_mats = []

            break_flag = 0
            for idx, midi in enumerate(track_midis):
                track_name = track_names[idx].replace('.mid', '')
                meta = metadata[track_name]
                if meta['is_drum']:  # Skip drum for now
                    continue
                elif not meta['audio_rendered']:  # some tracks are not rendered
                    continue
                else:
                    track_name_list.append(track_name)
                    track_piano_roll, track_program, track_quantize_error = midi2matrix(midi, pos_16th)
                    if track_quantize_error[0] > .2:
                        break_flag = 1
                        break
                    all_piano_rolls.append(track_piano_roll[..., 0])
                    dynamic_mats.append(track_piano_roll[..., 1])
                    # onset_mats.append(track_tpd_mat[..., 2])
                    # offset_mats.append(track_tpd_mat[..., 3])
                    programs.append(meta['program_num'])
            if break_flag:
                continue  # Skip pieces with large quantization error. This pieces are possibly triple-quaver songs

            # Concat all tpd mat from different tracks of a song together
            all_piano_rolls = np.concatenate(all_piano_rolls, axis=0)
            programs = np.array(programs)
            dynamic_mats = np.concatenate(dynamic_mats, axis=0)

            # onset_mats = np.concatenate(onset_mats, axis=0)
            # offset_mats = np.concatenate(offset_mats, axis=0)

            downbeat_indicator = np.array([int(t in downbeats) for t in pos_16th])

            np.savez_compressed(
                os.path.join(split_save_path, f'{song}.npz'),
                track_names=track_name_list,
                tracks=all_piano_rolls,
                programs=programs,
                db_indicator=downbeat_indicator,
                dynamics=dynamic_mats,
                pos_16th=pos_16th,
            )


def test_get_proll_from_mix():
    midi_fp = '../local/mt3_transcribed.mid'
    quantized_piano_roll = get_piano_rolls_and_other_features_from_mix_midi(midi_fp)


def quantize_drum(drum_midi_fp, pos_16th):
    '''
    Obtain the quantized piano roll for the drum track
    '''
    midi = pyd.PrettyMIDI(drum_midi_fp)
    track_piano_roll, track_program, track_quantize_error = midi2matrix(midi, pos_16th)
    return track_piano_roll


def get_piano_rolls_and_other_features_from_mix_midi(midi_fp, pos_16th_in_sec):
    '''
    Obtain all useful information for training, e.g., piano roll, without using track midis
    :param midi_fp:
    :return:
        If success, return a dictionary indexed by instrument name, with quantized piano roll as values
        If failed, return None.
    '''
    pos_per_beat = 4  # quantize each beat as 4 positions

    # Read a song with multiple tracks
    mixture_midi = pyd.PrettyMIDI(midi_fp)

    # Process only 2/4 and 4/4 songs
    for ts in mixture_midi.time_signature_changes:
        if ts.denominator == 4 and (ts.numerator == 2 or ts.numerator == 4):
            continue
        else:
            return None

    start_time = 0
    beats = mixture_midi.get_beats(start_time=start_time)
    downbeats = mixture_midi.get_downbeats(start_time=start_time)

    # Longshen: when dealing with MT3 output, need to use pre-defined pos of 16th note.
    # Get all 16th note position
    # beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))  # 补充上结束时间点
    # quantize = interp1d(np.arange(0, len(beats)) * pos_per_beat, beats, kind='linear')
    # pos_16th_in_sec = quantize(np.arange(0, (len(beats) - 1) * pos_per_beat))  # Get position of each

    # # Get the starting index of each bar
    # downbeat_indicator = np.array([int(t in downbeats) for t in pos_16th_in_sec])

    inst_util = InstMapUtil()
    quantized_piano_rolls = {}
    for track in mixture_midi.instruments:
        # Longshen did not delete the drum track this time
        if track.is_drum == True:
            program_num = 128
            pass
        else:
            program_num = track.program
        slakh_id, inst_name = inst_util.slakh_from_midi_program_get_id_and_inst(program_num)
        quantized_piano_roll, qt_error = inst2matrix(track, pos_16th_in_sec)  # Note that when fail, will return None
        quantized_piano_rolls[inst_name] = quantized_piano_roll

        # # Longshen: it's not necessary to check the qt error. There should be a lot of errors in the transcribed results
        # if qt_error > .2:
        #     b=2
        #     return None  # Skip pieces with large quantization error. This pieces are possibly triple-quaver songs

    # Longshen: also not necessary to return the time in sec, nor the bar start indicator
    return quantized_piano_rolls


def midi2matrix(midi, pos_16th):
    """
    Convert multi-track midi to a 3D matrix of shape (Track, Time, pitch=128, dur/vel=2).
    Each cell is an integer number representing quantized duration.
    :return
        Time-Pitch-Duration/velocity/CC matrics of each track (ndarray[#track, #pos, 128, 3])
        The program number of each track (list)

    """
    tpd_mats = []
    programs = []
    quant_errors = []
    tolerance = (pos_16th[1] - pos_16th[0]) / 2  # 如果音符比第一个downbeat还早，不计入
    for track in midi.instruments:
        qt_error = []  # record quantization error
        tpd_mat = np.zeros((len(pos_16th), 128, 2))  # time-pitch-duration
        for note in track.notes:
            if note.start < pos_16th[0] - tolerance:
                continue
            note_start = np.argmin(
                np.abs(pos_16th - note.start))  # The closest 16th position to the note start time, start from 0
            note_end = np.argmin(np.abs(pos_16th - note.end))  # The closest 16th position to the note end time
            if note_end == note_start:
                note_end = min(note_start + 1, len(pos_16th) - 1)  # guitar/bass plunk typically results in
                # a very short note duration. These note should be quantized to 1 instead of 0.
            tpd_mat[note_start, note.pitch, 0] = note_end - note_start  # duration
            tpd_mat[note_start, note.pitch, 1] = note.velocity  # velocity
            # tpd_mat[note_start, note.pitch, 2] = note.start               # onset
            # tpd_mat[note_start, note.pitch, 3] = note.end                 # offset

            # Compute quantization error. A song with very high error (e.g., triple-quaver songs) will be discriminated and therefore discarded.
            if note_end == note_start:
                qt_error.append(
                    np.abs(pos_16th[note_start] - note.start) / (pos_16th[note_start] - pos_16th[note_start - 1]))
            else:
                qt_error.append(
                    np.abs(pos_16th[note_start] - note.start) / (pos_16th[note_end] - pos_16th[note_start]))

        control_matrix = np.ones((len(pos_16th), 128, 1)) * -1
        for control in track.control_changes:
            # if control.time < time_end:
            #    if len(quaver) == 0:
            #        continue
            control_time = np.argmin(np.abs(pos_16th - control.time))
            control_matrix[control_time, control.number, 0] = control.value

        tpd_mat = np.concatenate((tpd_mat, control_matrix), axis=-1)
        tpd_mats.append(tpd_mat)
        programs.append(track.program)
        quant_errors.append(np.mean(qt_error))

    tpd_mats = np.array(tpd_mats)
    return tpd_mats, programs, quant_errors


def inst2matrix(inst, pos_16th_in_sec):
    """
    Convert a prettymidi instrument track to a 2D matrix of quantized piano roll
    (Time, pitch=128, dur/vel=2).
    Each cell is an integer number representing quantized duration.
    :return
        Time-Pitch-Duration/velocity/CC matrics of each track (ndarray[#track, #pos, 128, 3])
        The program number of each track (list)
    """
    tolerance = (pos_16th_in_sec[1] - pos_16th_in_sec[0]) / 2  # 如果音符比第一个downbeat还早，不计入

    qt_error = []  # record quantization error
    quantized_piano_roll = np.zeros((len(pos_16th_in_sec), 128))  # time-pitch-duration
    for note in inst.notes:
        if note.start < pos_16th_in_sec[0] - tolerance:
            continue
        note_start = np.argmin(
            np.abs(pos_16th_in_sec - note.start))  # The closest 16th position to the note start time, start from 0
        note_end = np.argmin(np.abs(pos_16th_in_sec - note.end))  # The closest 16th position to the note end time
        if note_end == note_start:
            note_end = min(note_start + 1, len(pos_16th_in_sec) - 1)  # guitar/bass plunk typically results in
            # a very short note duration. These note should be quantized to 1 instead of 0.
        quantized_piano_roll[note_start, note.pitch] = note_end - note_start  # duration

        # Compute quantization error. A song with very high error (e.g., triple-quaver songs) will be discriminated and therefore discarded.
        if note_end == note_start:
            qt_error.append(
                np.abs(pos_16th_in_sec[note_start] - note.start) / (
                        pos_16th_in_sec[note_start] - pos_16th_in_sec[note_start - 1]))
        else:
            qt_error.append(
                np.abs(pos_16th_in_sec[note_start] - note.start) / (
                        pos_16th_in_sec[note_end] - pos_16th_in_sec[note_start]))

    return quantized_piano_roll, np.mean(qt_error)


if __name__ == '__main__':
    _main()

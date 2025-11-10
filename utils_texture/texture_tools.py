import os
import sys

import numpy as np

def _main():
    pass


def _procedures():
    pass

def get_onset_density_of_a_bar_from_remi(remi_seq):
    '''
    Count the number of onset, for all non-empty position of a bar.
    '''
    t = {}
    for i, tok in enumerate(remi_seq):
        if tok == 'b-1':
            assert i == len(remi_seq) - 1
        elif tok.startswith('o-'):
            cur_pos = int(tok.split('-')[-1])
            t[cur_pos] = 0
        elif tok.startswith('p-'):
            t[cur_pos] += 1
    return t

def tokenize_onset_density_one_bar(onset_dict: dict, quantize=False):
    '''
    Convert an onset density dict to a list of token sequence
    Quantize: Binarize onset count. If count > quantile(0.5), set value as 2, otherwise 1.
    '''
    if len(onset_dict) == 0:
        return []

    # Quantize the onset count
    if quantize:
        t = list(onset_dict.values())
        median = np.quantile(t, 0.5)
        for k in onset_dict:
            if onset_dict[k] > median:
                onset_dict[k] = 2
            else:
                onset_dict[k] = 1
    
    # Tokenization
    res = []
    for pos in onset_dict:
        onset_cnt = onset_dict[pos]
        res.append('o-{}'.format(pos))
        res.append('TF-{}'.format(onset_cnt))
    return res


def get_time_function_from_remi_one_bar(remi_seq, return_dict=False):
    res = ['TF']

    note_this_pos = 0
    for token in remi_seq:
        if token.startswith('o-'):
            # Add note count of previous position to result
            if note_this_pos > 0:
                res.append('TF-{}'.format(note_this_pos))

            # Start counting for new position
            res.append(token)
            note_this_pos = 0
        elif token.startswith('p-'):
            # Count the note
            note_this_pos += 1

    # At the end of sequence, add the note count of the last position
    if note_this_pos > 0:
        res.append('TF-{}'.format(note_this_pos))

    if return_dict:
        seq = res
        res = []
        t = {}
        for tok in seq:
            if tok == 'b-1':
                res.append(t)
                t = {}
            elif tok.startswith('o-'):
                cur_pos = int(tok.split('-')[-1])
            elif tok.startswith('TF-'):
                onset_cnt = int(tok.split('-')[-1])
                t[cur_pos] = onset_cnt
        return res

    return res


def get_time_function_from_remi_multiple_bars(remi_seq):
    '''
    Get time function info from remi of multiple bars.
    Return a list of dict, each dict is info of one bar.
    '''

    res = []
    t = {}
    note_this_pos = 0
    cur_pos = 0
    for token in remi_seq:
        if token == 'b-1':
            if note_this_pos > 0:
                t[cur_pos] = note_this_pos
            res.append(t)
            t = {}
            note_this_pos = 0
        if token.startswith('o-'):
            # Add note count of previous position to result
            if note_this_pos > 0:
                t[cur_pos] = note_this_pos

            # Start counting for new position
            cur_pos = int(token.split('-')[-1])
            note_this_pos = 0
        elif token.startswith('p-'):
            # Count the note
            note_this_pos += 1
    #
    # # At the end of sequence, add the note count of the last position
    # if note_this_pos > 0:
    #     res.append('TF-{}'.format(note_this_pos))
    #
    # for tok in inp_seq:
    #     if tok == 'TF':
    #         t = {}
    #     elif tok == 'b-1':
    #         res.append(t)
    #     elif tok.startswith('o-'):
    #         cur_pos = int(tok.split('-')[-1])
    #     elif tok.startswith('TF-'):
    #         onset_cnt = int(tok.split('-')[-1])
    #         t[cur_pos] = onset_cnt
    return res

def get_pitch_seq_from_input(inp_seq):
    '''
    Obtain pitch sequence from input token sequence
    Return a list of list. Each sublist is the pitch sequence of one bar.
    '''
    res = []
    
    for tok in inp_seq:
        if tok == 'PITCH':
            cur_bar = []
        elif tok.startswith('p-'):
            cur_bar.append(tok)
        elif tok == 'TF':
            res.append(cur_bar)
            cur_bar = []
    return res

def get_pitch_seq_from_remi(remi_seq):
    '''
    len(return) = #bars
    '''
    pitch_seqs = []
    from utils_midi.utils_midi import RemiUtil

    b_1_indices = RemiUtil.get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)

    if num_bars == 0:
        print('bar num = 0')
        return []
        raise Exception("Bar num = 0")

    # Iterate over all bars
    for bar_id in b_1_indices:
        bar_start_idx, bar_end_idx = b_1_indices[bar_id]
        bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

        pitch_tokens = []

        """ Only retain position, pitch, and bar line """
        for tok in bar_remi_seq:
            if tok.startswith("p-"):
                pitch_tokens.append(tok)

        pitch_seqs.append(pitch_tokens)

    return pitch_seqs


# def get_time_function_from_input_seq():


if __name__ == '__main__':
    _main()

'''
Objective evaluation for piano arrangement (reduction)
'''
import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(__file__)))

from remi_z import MultiTrack, Bar  
from evaluations.general import calculate_wer
import jiwer
from sklearn.metrics import f1_score
import numpy as np
import torch


def bar_level_pitch_wer_from_proll(proll_out:torch.Tensor, proll_ref:torch.Tensor):
    '''
    Compute the pitch WER for a segment-level piano arrangement
    
    '''
    proll_out = proll_out.cpu().numpy()
    proll_ref = proll_ref.cpu().numpy()

    # Binarize
    proll_ref = (proll_ref > 0).astype(int) # [16, 128]
    proll_out = (proll_out > 0).astype(int)

    # Prepare pitch seq
    tgt_seq = []
    for pos, pitch in zip(proll_ref.nonzero()[0], proll_ref.nonzero()[1]):
        tgt_seq.append(pitch)
    out_seq = []
    for pos, pitch in zip(proll_out.nonzero()[0], proll_out.nonzero()[1]):
        out_seq.append(pitch)

    # Calculate the pitch WER
    tgt_p_seq = [str(n) for n in tgt_seq]
    out_p_seq = [str(n) for n in out_seq]
    tgt_str = ' '.join(tgt_p_seq)
    out_str = ' '.join(out_p_seq)

    # Calculate the pitch WER
    wer = jiwer.wer(tgt_str, out_str)
    return wer


def bar_level_pos_wer_from_proll(proll_out:torch.Tensor, proll_ref:torch.Tensor):
    '''
    Compute the position WER for a segment-level piano arrangement
    '''
    proll_out = proll_out.cpu().numpy()
    proll_ref = proll_ref.cpu().numpy()

    # Binarize
    proll_ref = (proll_ref > 0).astype(int) # [16, 128]
    proll_out = (proll_out > 0).astype(int)

    # Get non-zero position
    pos_ref = proll_ref.nonzero()[0]
    pos_out = proll_out.nonzero()[0]
    pos_ref = list(set(pos_ref))
    pos_out = list(set(pos_out))
    pos_ref.sort()
    pos_out.sort()

    # Make them to pitch sequences
    tgt_p_seq = [str(n) for n in pos_ref]
    out_p_seq = [str(n) for n in pos_out]
    tgt_str = ' '.join(tgt_p_seq)
    out_str = ' '.join(out_p_seq)

    # Calculate the pitch WER
    wer = jiwer.wer(tgt_str, out_str)
    return wer


def bar_level_note_f1_from_proll(proll_out:torch.Tensor, proll_ref:torch.Tensor):
    '''
    Compute the note F1 for a segment-level piano arrangement
    '''
    proll_out = proll_out.cpu().numpy()
    proll_ref = proll_ref.cpu().numpy()

    # Binarize
    proll_ref = (proll_ref > 0).astype(int) # [16, 128]
    proll_out = (proll_out > 0).astype(int)

    # Flatten
    proll_ref = proll_ref.flatten()
    proll_out = proll_out.flatten()

    # Calculate the note F1
    f1 = f1_score(proll_ref, proll_out)
    return f1


def song_level_pitch_wer(out_midi_fp, ref_midi_fp):
    '''
    Compute the pitch WER for a song-level piano arrangement
    '''
    # Load the MIDI files
    ref_mt = MultiTrack.from_midi(ref_midi_fp)
    out_mt = MultiTrack.from_midi(out_midi_fp)

    # Get piano notes from MultiTrack
    piano_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tgt_seq = ref_mt.get_all_notes(include_drum=False, of_insts=piano_ids)
    out_seq = out_mt.get_all_notes(include_drum=False, of_insts=piano_ids)

    # Make them to pitch sequences
    tgt_p_seq = [str(n.pitch) for n in tgt_seq]
    out_p_seq = [str(n.pitch) for n in out_seq]
    tgt_str = ' '.join(tgt_p_seq)
    out_str = ' '.join(out_p_seq)

    # Calculate the pitch WER
    wer = jiwer.wer(tgt_str, out_str)
    # pitch_wer = calculate_wer(out_seq, tgt_seq)
    return wer

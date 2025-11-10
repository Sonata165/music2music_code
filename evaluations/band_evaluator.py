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



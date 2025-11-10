import os
import sys

sys.path.append('..')

from utils_midi.utils_midi import RemiTokenizer

def convert_remi_to_midi():
    tk = RemiTokenizer()
    remi_fp = '../_misc/out.txt'
    midi_fp = '../_misc/out.mid'
    tk.remi_file_to_midi(remi_fp=remi_fp, midi_path=midi_fp)

if __name__ == '__main__':
    convert_remi_to_midi()
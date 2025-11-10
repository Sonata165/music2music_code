'''
Tokenize and detokenize the midi file
'''

import os
import sys
sys.path.append('..')

from utils_midi.utils_midi import RemiTokenizer
from utils_common.utils import jpath, read_yaml

def main():
    midi_fp = "/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/infer_input/full_song/slakh/demo drum/all_src1887.mid"
    prepare_midi(midi_fp)

def prepare_midi(midi_fp):
    tk = RemiTokenizer()
    midi_dir = os.path.dirname(midi_fp)
    remi_fp = jpath(midi_dir, 'remi.txt')
    new_midi_fp = midi_fp.replace('.mid', '_norm.mid')

    tk.midi_to_remi_file(midi_fp, remi_fp)
    tk.remi_file_to_midi(remi_fp, new_midi_fp)

if __name__ == '__main__':
    main()
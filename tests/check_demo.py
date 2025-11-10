import os
import sys

sys.path.append('..')

from utils_midi.utils_midi import RemiTokenizer
from utils_midi import remi_utils

def convert_midi_to_remi():
    tk = RemiTokenizer()
    midi_fp = '../_misc/remiz_demo.mid'
    remi_fp = '../_misc/remiz_demo.txt'
    
    # tk.remi_file_to_midi(remi_fp=remi_fp, midi_path=midi_fp)
    remi = tk.midi_to_remi(midi_path=midi_fp, normalize_pitch=False, return_pitch_shift=False, return_key=False,
                    reorder_by_inst=True, include_ts=False, include_tempo=False, include_velocity=False)
    
    content = remi_utils.from_remi_mbar_get_global_opd_seq(remi)

    remi_str = ' '.join(content)
    with open(remi_fp, 'w') as f:
        f.write(remi_str)

# def convert_remi_to_midi():
#     tk = RemiTokenizer()
#     remi_fp = '../_misc/out.txt'
#     midi_fp = '../_misc/out.mid'
#     tk.remi_file_to_midi(remi_fp=remi_fp, midi_path=midi_fp)

if __name__ == '__main__':
    # convert_remi_to_midi()
    convert_midi_to_remi()
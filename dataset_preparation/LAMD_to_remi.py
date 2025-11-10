'''
Preprocess the LAMD dataset to REMI format for training
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('musecoco')

from utils_midi.utils_midi import RemiTokenizer
from src_hf.utils import jpath, ls
from tqdm import tqdm


def main():
    clean_remi()
    pass

def midi_to_remi_all():
    data_dir = '/data2/longshen/Datasets/LAMD_v4/LAMD/MIDIs'
    remi_dir = '/data2/longshen/Datasets/LAMD_v4/LAMD/REMI'

    tk = RemiTokenizer()
    
    dir_names = ls(data_dir)
    error_rate = 0
    for dir_name in dir_names:
        dir_path = jpath(data_dir, dir_name)
        midi_fns = ls(dir_path)
        pbar = tqdm(midi_fns)
        cnt = 0
        error_cnt = 0
        for midi_fn in pbar:
            pbar.set_description('Processing {} {}, error rate: {:.2f}%'.format(dir_name, midi_fn, error_rate))
            cnt += 1
            try:
                midi_fp = jpath(dir_path, midi_fn)
                out_fp = jpath(remi_dir, dir_name, midi_fn.replace('.mid', '.txt'))

                out = tk.midi_to_remi(midi_fp, normalize_pitch=False)

                remi_tok_strs = ' '.join(out)
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)
                with open(out_fp, 'w') as f:
                    f.write(remi_tok_strs + '\n')
            except:
                error_cnt += 1
                continue

            if cnt % 100 == 0:
                error_rate = 100 * error_cnt / cnt   

def clean_remi():

    a = sys.argv[1]
    print(a)
    # exit(10)

    remi_dir = '/data2/longshen/Datasets/LAMD_v4/LAMD/REMI'
    remi_organized_dir = '/data2/longshen/Datasets/LAMD_v4/LAMD/REMI_ORGANIZED'
    from src_hf.utils import jpath, ls, create_dir_if_not_exist
    create_dir_if_not_exist(remi_organized_dir)

    from utils_midi import remi_utils

    tk = RemiTokenizer()
    
    dir_names = ls(remi_dir)
    error_rate = 0
    for dir_name in dir_names:
        
        dir_name = a

        dir_path = jpath(remi_dir, dir_name)
        remi_fns = ls(dir_path)
        pbar = tqdm(remi_fns)
        cnt = 0
        error_cnt = 0
        for remi_fn in pbar:
            pbar.set_description('Processing {} {}, error rate: {:.2f}%'.format(dir_name, remi_fn, error_rate))
            cnt += 1
            try:
                remi_fp = jpath(dir_path, remi_fn)
                out_fp = jpath(remi_organized_dir, dir_name, remi_fn)
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)

                with open(remi_fp, 'r') as f:
                    remi_str = f.read()
                
                remi_seq = remi_str.strip().split()

                # Do something
                bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
                organized_remi = []
                for bar_id, (bar_start_idx, bar_end_idx) in bar_indices.items():
                    bar_seq = remi_seq[bar_start_idx:bar_end_idx]
                    new_bar = remi_utils.reorder_remi_bar(bar_seq, add_bar_token=True)
                    organized_remi.extend(new_bar)
                    # remi_utils.save_remi(bar_seq, jpath(out_fp, f'{bar_id}.txt'))

                    # Quantize the instrument
                
                bar_indices_new = remi_utils.from_remi_get_bar_idx(organized_remi)
                if len(bar_indices) != len(bar_indices_new):
                    raise ValueError('Error in bar indices')

                remi_tok_strs = ' '.join(organized_remi)
                with open(out_fp, 'w') as f:
                    f.write(remi_tok_strs + '\n')
            except:
                error_cnt += 1
                continue

            if cnt % 100 == 0:
                error_rate = 100 * error_cnt / cnt   
        
        return


if __name__ == '__main__':
    main()
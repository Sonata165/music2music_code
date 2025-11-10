import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from src_hf.utils import jpath

def main():
    check_non_inst_note()

def check_non_inst_note():
    '''
    There are some cases where the first note of a bar does not have instrument information.
    Count such cases.

    split: valid, cnt: 6
    split: test, cnt: 4
    split: train, cnt: 331

    So ignore these cases.
    '''
    data_root = '/data2/longshen/musecoco_data/datasets'
    splits = ['valid', 'test', 'train']
    for split in splits:
        data_fn = '{}.txt'.format(split)
        data_fp = jpath(data_root, data_fn)
        with open(data_fp, 'r') as f:
            lines = f.readlines()
        cnt = 0
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 2:
                continue

            first_bar_line_idx = line.index('b-1')
            first_tok_of_second_bar_index = first_bar_line_idx + 2
            if first_tok_of_second_bar_index >= len(line):
                continue

            if not line[first_bar_line_idx + 2].startswith('i'):
                cnt += 1
        print('split: {}, cnt: {}'.format(split, cnt))

if __name__ == '__main__':
    main()
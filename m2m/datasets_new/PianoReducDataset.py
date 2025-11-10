import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(dirof(__file__))))

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils_midi import remi_utils
from typing import List
from tqdm import tqdm
from utils_instrument.inst_map import InstMapUtil
from utils_chord.chord_detect_from_remi import chord_to_id
from utils_common.utils import read_json
from remi_z import MultiTrack, Bar


class PianoReducDataset(Dataset):
    '''
    Piano retrieval dataset for reduction
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config):
        # Read the bar-level dataset (single json)
        split_name = split if split != 'valid' else 'validation'
        ori_data = read_json(data_fp)[split_name]
        ori_size = len(ori_data)
        self.ori_data = ori_data # Save the original data

        # Filter out data that does not have piano in the target sequence
        if split != 'infer':
            print('Filtering data ...')
            data_new = {}
            for bar_name, bar_info in ori_data.items():
                meta = bar_info['meta']
                is_empty = meta['pitch_range'] == -1

                # Filter out non-4/4 time signature
                if meta['time_signature'] != '(4, 4)':
                    continue

                # Can be an empty bar in training set
                if is_empty and split == 'train':
                    pass
                else:
                    # If not empty, need to have piano
                    if meta['has_piano'] is False:
                        continue

                    # Piano range >= 0.4 * total pitch range
                    piano_prange = meta['piano_pitch_range']
                    total_prange = meta['pitch_range']
                    if piano_prange < 0.4 * total_prange:
                        continue

                data_new[bar_name] = bar_info

            data = data_new
            filtered_size = len(data)
            filter_rate = filtered_size / ori_size
            print(f'Filtered {ori_size - filtered_size} samples, {filter_rate:.2f} left')
        else:
            data = ori_data
            print('No filtering for inference data')
            
        # Re-index with integer starting from 0
        idx = 0
        data_indexed = {}
        for bar_name in data:
            entry = data[bar_name]
            entry['bar_name'] = bar_name
            data_indexed[idx] = entry
            idx += 1
        self.data = data_indexed

        self.split = split
        self.config = config
        self.piano_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]

        # Get bar remi-z
        tgt_str = data_entry['content']

        # Get content seq
        bar = MultiTrack.from_remiz_str(tgt_str)[0]
        content_seq = bar.get_content_seq(
            include_drum=False,
            with_dur=True,
        )

        # Get history remi-z
        hist_bar_name = data_entry['hist'] # Can be None or a bar name
        if hist_bar_name is None:
            hist_str = ''
        else:
            hist_str = self.ori_data[hist_bar_name]['content']
        
        # Get history piano seq
        if hist_str == '':
            hist_piano_content_seq = []
        else:
            # print(hist_bar_name)
            hist_bar = MultiTrack.from_remiz_str(hist_str)[0]
            hist_piano_content_seq = hist_bar.get_content_seq(
                include_drum=False,
                of_insts=self.piano_ids,
                with_dur=True,
            )

        # Obtain piano remi-z
        piano_remiz = bar.get_content_seq(
            include_drum=False,
            of_insts=self.piano_ids,
            with_dur=True,
        )

        # Get instrument
        insts = ['i-0']

        ''' Condition format: [BOS] [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_piano_content_seq + ['[SEP]'] + piano_remiz + ['[EOS]']
        inp_str = ' '.join(inp_seq)
        
        return inp_str
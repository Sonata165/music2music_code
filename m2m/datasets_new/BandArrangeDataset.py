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


class BandArrangeDataset(Dataset):
    '''
    Band arrangement dataset
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config):
        # Read the bar-level dataset (single json)
        split_name = split if split != 'valid' else 'validation'
        ori_data = read_json(data_fp)[split_name]
        ori_size = len(ori_data)
        self.ori_data = ori_data # Save the original data

        # Filter data: remove non-4/4 time signature, empty bars (valid and test)
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
                if is_empty:
                    if split == 'train':
                        pass
                    else:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]

        # Get bar object
        tgt_str = data_entry['content']
        bar = MultiTrack.from_remiz_str(tgt_str)[0]

        # Instrument quantization
        inst_map = InstMapUtil()
        bar = quantize_inst_for_bar(bar, inst_map)

        # Get target seq
        if self.config.get('use_remi_plus', False):
            tgt_seq = bar.to_remiplus_seq(
                with_ts=False,
                with_tempo=False,
                with_velocity=False,
                include_drum=False,
            )
        else:
            tgt_seq = bar.to_remiz_seq(
                with_ts=False,
                with_tempo=False,
                with_velocity=False,
                include_drum=False,
            )

        # Get content seq
        content_seq = bar.get_content_seq(
            include_drum=False,
            with_dur=False,
        )

        # Get history bar
        hist_bar_name = data_entry['hist'] # Can be None or a bar name
        if hist_bar_name is None:
            hist_str = ''
        else:
            hist_str = self.ori_data[hist_bar_name]['content']
        
        # Get history remi-z seq
        if hist_str == '':
            hist_seq = []
        else:
            # print(hist_bar_name)
            hist_bar = MultiTrack.from_remiz_str(hist_str)[0]
            hist_bar = quantize_inst_for_bar(hist_bar, inst_map)

            if self.config.get('use_remi_plus', False):
                hist_seq = hist_bar.to_remiplus_seq(
                    with_ts=False,
                    with_tempo=False,
                    with_velocity=False,
                    include_drum=False,
                )
            else:
                hist_seq = hist_bar.to_remiz_seq(
                    with_ts=False,
                    with_tempo=False,
                    with_velocity=False,
                    include_drum=False,
                )

        # Get instruments
        inst_ids = bar.get_unique_insts(sort_by_voice=True, include_drum=False)
        if self.config.get('no_voice', False):
            inst_ids.sort()
        insts = [f'i-{inst_id}' for inst_id in inst_ids]

        if self.config.get('no_hist', False):
            hist_seq = []

        ''' Condition format: [BOS] [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        inp_str = ' '.join(inp_seq)
        
        return inp_str


def quantize_inst_for_bar(bar: Bar, inst_map: InstMapUtil):
    '''
    Quantize the instrument program number for a bar

    NOTE: In-place operation
    '''
    # Quantize instrument program ID for each track
    track_dict = {}
    for track_id, track in bar.tracks.items():
        inst_id_new = inst_map.slakh_quantize_inst_prog(track_id)
        if inst_id_new is None: # Skip the track if the instrument is not supported
            continue
        track.set_inst_id(inst_id_new)

        if inst_id_new not in track_dict:
            track_dict[inst_id_new] = track
        else:
            track_dict[inst_id_new].merge_with(track)    
    track_list = list(track_dict.values())

    # Create a new bar object with new tracks
    bar_new = Bar.from_tracks(
        bar_id = bar.bar_id,
        track_list=track_list,
        time_signature=bar.time_signature,
        tempo=bar.tempo,
    )

    return bar_new
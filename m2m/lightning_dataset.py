import os
import sys
dirof = os.path.dirname
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, dirof(dirof(__file__)))

import random
import torch
from torch import utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils_midi import remi_utils
from typing import List
from tqdm import tqdm
from utils_instrument.inst_map import InstMapUtil
from utils_chord.chord_detect_from_remi import chord_to_id
from utils_common.utils import jpath
from m2m.datasets_new.PianoReducDataset import PianoReducDataset
from m2m.datasets_new.BandArrangeDataset import BandArrangeDataset
from m2m.datasets_new.UnconditionalDataset import UnconditionalDataset


def get_dataloader(config, split):
    '''
    Only used by lightning_train_new.py
    '''
    bs = config['bs'] if split != 'test' else config['bs_test']
    
    data_fp = config['data_root']

    dataset_class_name = config['dataset_class']
    dataset_class = eval(dataset_class_name)

    dataset = dataset_class(data_fp=data_fp, split=split, config=config)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=bs,
        num_workers=config['num_workers'],
        shuffle=True if split == 'train' else False,
        collate_fn=lambda x: x,
    )
    return dataloader


class FixedBarGenDataset(Dataset):
    '''
    The dataset class for fixed-length 8-bar conditional generation 
    Conditions:
        - (8-bar length)
        - Time signature
        - Tempo
        - Instrument (with voice control)
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.data = data
        self.split = split

        self.voice_control = config['voice_control'] if 'voice_control' in config else False
        self.texture_control = config['texture_control'] if 'texture_control' in config else False
        self.aug_hist = config['aug_hist'] if 'aug_hist' in config else False

        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')

        # Get time signature
        ts = remi_seq[0]

        # Get tempo
        tempo = remi_seq[1]

        # Get instrument
        insts = remi_utils.from_remi_get_inst_and_voice(remi_seq)

        # Target sequence
        tgt_seq = remi_seq

        ''' Data Augmentation for instrument control '''
        if self.config['inst_aug'] is True and self.split == 'train':
            # Randomly delete some tracks
            tgt_seq = remi_utils.in_remi_multi_bar_delete_insts(tgt_seq)

            # Re-obtain the instrument list
            insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)

        if 'numbered_bar' in self.config and self.config['numbered_bar'] is True:
            # Replace the bar line token from b-1 only to b-1, b-2, ...
            tgt_seq = remi_utils.in_remi_multi_bar_replace_bar_tokens(tgt_seq)

        ''' Sequence format: [BOS] [time_signature] [tempo] [INST] [insts] [SEP] [tgt_remi] [EOS] '''
        # Assembly the input sequence
        input_seq = []
        input_seq.append('[BOS]')
        # input_seq.append(ts)
        # input_seq.append(tempo)
        input_seq.append('[INST]')
        input_seq.extend(insts)
        # input_seq.append('[PITCH]')
        input_seq.append('[SEP]')
        
        if self.split != 'test':
            input_seq.extend(tgt_seq)
            input_seq.append('[EOS]')
        
        input_str = ' '.join(input_seq)

        return input_str



class SourceSepDataset(Dataset):
    '''
    The dataset class for 2-bar symbolic source separation

    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.data = data
        self.split = split
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')
       
        # Target-side sequence (reordered by instrument)
        tgt_seq = remi_seq

        # Get content (pos and pitch sequence)
        bar_1, bar_2 = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)
        content_seq_tot = []
        for bar in [bar_1, bar_2]:
            content_seq = remi_utils.from_remi_get_pitch_of_pos_seq(bar, flatten=False)
            # content_seq = remi_utils.from_remi_get_global_opd_seq(bar)
            content_seq_tot.extend(content_seq)
            content_seq_tot.append('b-1')

        # Get instrument (no voice info)
        if self.config.get('voice_control') is True:
            insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        else:
            insts = remi_utils.from_remi_get_insts(tgt_seq)

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq_tot + ['[SEP]'] + tgt_seq + ['[EOS]']

        inp_str = ' '.join(inp_seq)
        
        return inp_str
    


class SourceSepDatasetRemiP(Dataset):
    '''
    The dataset class for 2-bar symbolic source separation

    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence

    In REMI+ format
    '''
    def __init__(self, data_fp, split, config):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.data = data
        self.split = split
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')
       
        # Target-side sequence (reordered by instrument)
        tgt_seq = remi_seq

        # Get content (pos and pitch sequence)
        bar_1, bar_2 = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)
        content_seq_tot = []
        for bar in [bar_1, bar_2]:
            content_seq = remi_utils.from_remi_get_pitch_of_pos_seq(bar, flatten=False) # No duration in content
            # content_seq = remi_utils.from_remi_get_global_opd_seq(bar)
            content_seq_tot.extend(content_seq)
            content_seq_tot.append('b-1')

        # Get instrument (no voice info)
        if self.config.get('voice_control') is True:
            insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        else:
            insts = remi_utils.from_remi_get_insts(tgt_seq)

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Convert tgt_seq to REMI+ format
        tgt_seq = remi_utils.from_remi_z_to_remi_plus(tgt_seq)

        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq_tot + ['[SEP]'] + tgt_seq + ['[EOS]']

        inp_str = ' '.join(inp_seq)
        
        return inp_str



class ArrangerDataset(Dataset):
    '''
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.data = data

        self.split = split
        self.config = config
        self.rand_inst_infer = rand_inst_infer

        self.remi_aug = ArrangerAugment(config=config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')
       
        # Pitch shift augmentation
        if self.config.get('pitch_shift', False) is True and self.split == 'train':
            remi_seq = remi_utils.in_remi_shift_pitch_random(remi_seq)

        # Prepare hist_seq, tgt_seq, tgt_for_content
        if self.config.get('content_consistent', False) is True:
            # Prepare a copy of the remi sequence for content augmentation
            remi_seq_for_content = remi_seq.copy()

            # Content augmentation
            if self.config.get('content_aug') is True and self.split == 'train':
                remi_seq_for_content = remi_utils.in_remi_multi_bar_delete_non_mel_insts(remi_seq_for_content)

            # Hist and target sequence (augmented)
            hist_seq, tgt_for_content = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq_for_content)

            # True target sequence
            _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)
        else:
            # Hist and target sequence
            hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

            # Prepare a copy of target sequence for content sequence
            tgt_for_content = tgt_seq.copy()
            if self.config.get('content_aug') is True and self.split == 'train':
                tgt_for_content = remi_utils.in_remi_multi_bar_delete_non_mel_insts(tgt_for_content)
        
        # Prepare content sequence from target (copied)
        if self.config.get('dur_input') is True:
            content_seq = remi_utils.from_remi_get_global_opd_seq(tgt_for_content)
        else:
            content_seq = remi_utils.from_remi_get_pitch_of_pos_seq(tgt_for_content, flatten=True)

        # Target-side and instrument spec augmentation (melody preserve)
        if self.config['inst_aug'] is True and self.split in ['train']:
            # Randomly delete some tracks
            tgt_seq = remi_utils.in_remi_multi_bar_delete_non_mel_insts(tgt_seq)

        # Get instrument
        if self.config.get('texture_control') is True:
            insts = remi_utils.from_remi_get_inst_voice_texture(tgt_seq)
        else:
            insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        inp_str = ' '.join(inp_seq)
        
        return inp_str
    


class RetrievalDatasetSrcHist(Dataset):
    '''
    Piano retrieval dataset for reduction
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip().split() for l in data] # a list of strings

        # Filter out data that does not have piano in the target sequence
        if split != 'infer':
            print('Filtering data ...')
            data_new = []
            for sample in data:
                _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)
                # Need to have piano
                if 'i-0' not in tgt_seq and 'i-2' not in tgt_seq:
                    continue
                
                # Piano need to be polyphonic
                ap_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
                ep_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
                t = {'i-0': ap_opd, 'i-2': ep_opd}
                piano_opd = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(t)
                pos_tok = [tok for tok in piano_opd if tok.startswith('o')]
                pitch_tok = [tok for tok in piano_opd if tok.startswith('p')]
                n_pos = len(pos_tok)
                n_pitch = len(pitch_tok)
                if n_pos == n_pitch:
                    continue

                data_new.append(sample)
            self.data = data_new
        else:
            self.data = data

        self.split = split
        self.config = config
        self.rand_inst_infer = rand_inst_infer

        self.remi_aug = ArrangerAugment(config=config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_seq = self.data[index]
       
        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

        # Get pos and pitch sequence
        tgt_for_content = tgt_seq.copy()
        if self.config.get('content_aug') is True and self.split == 'train':
            tgt_for_content = remi_utils.in_remi_multi_bar_delete_non_mel_insts(tgt_for_content)
        content_seq = remi_utils.from_remi_get_pitch_of_pos_seq(tgt_for_content, flatten=True)

        def get_piano_opd(tgt_seq):
            # Obtain 'i-0' and 'i-2' tracks, merge together, assign 'i-0' inst
            opd_dict = {}
            if 'i-0' in tgt_seq:
                ap_opd_seq = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
                opd_dict['i-0'] = ap_opd_seq
            if 'i-2' in tgt_seq:
                ep_opd_seq = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
                opd_dict['i-2'] = ep_opd_seq
            tgt_seq = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(opd_dict)
            if len(tgt_seq) > 0:
                tgt_seq = ['i-0'] + tgt_seq
            tgt_seq = tgt_seq + ['b-1']

            return tgt_seq
        
        # Get target sequence
        tgt_seq = get_piano_opd(tgt_seq)

        # # Modify history sequence
        # hist_seq = get_piano_opd(hist_seq)

        # Get instrument
        # insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        insts = ['i-0']

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        # if self.split != 'test':
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        # else: # Do not provide target when doing generation for test set
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        #     inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']

        inp_str = ' '.join(inp_seq)
        
        return inp_str


class P2BDataset(Dataset):
    '''
    Piano retrieval dataset for reduction
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip().split() for l in data] # a list of strings

        # Filter out data that does not have piano in the target sequence
        if split != 'infer':
            print('Filtering data ...')
            data_new = []
            for sample in data:
                _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)
                # Need to have piano
                if 'i-0' not in tgt_seq and 'i-2' not in tgt_seq:
                    continue
                
                # Piano need to be polyphonic
                ap_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
                ep_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
                t = {'i-0': ap_opd, 'i-2': ep_opd}
                piano_opd = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(t)
                pos_tok = [tok for tok in piano_opd if tok.startswith('o')]
                pitch_tok = [tok for tok in piano_opd if tok.startswith('p')]
                n_pos = len(pos_tok)
                n_pitch = len(pitch_tok)
                if n_pos == n_pitch:
                    continue

                data_new.append(sample)
            self.data = data_new
        else:
            self.data = data

        self.split = split
        self.config = config
        self.rand_inst_infer = rand_inst_infer

        self.remi_aug = ArrangerAugment(config=config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_seq = self.data[index]
       
        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

        # # Get pos and pitch sequence
        # tgt_for_content = tgt_seq.copy()
        # if self.config.get('content_aug') is True and self.split == 'train':
        #     tgt_for_content = remi_utils.in_remi_multi_bar_delete_non_mel_insts(tgt_for_content)
        
        # if self.config.get('dur_input') is True:
        #     content_seq = remi_utils.from_remi_get_global_opd_seq(tgt_for_content)
        # else:
        #     content_seq = remi_utils.from_remi_get_pitch_of_pos_seq(tgt_for_content, flatten=True)
        

        def get_piano_opd(tgt_seq):
            # Obtain 'i-0' and 'i-2' tracks, merge together, assign 'i-0' inst
            opd_dict = {}
            if 'i-0' in tgt_seq:
                ap_opd_seq = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
                opd_dict['i-0'] = ap_opd_seq
            if 'i-2' in tgt_seq:
                ep_opd_seq = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
                opd_dict['i-2'] = ep_opd_seq
            tgt_seq = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(opd_dict)
            if len(tgt_seq) > 0:
                tgt_seq = ['i-0'] + tgt_seq
            tgt_seq = tgt_seq + ['b-1']

            return tgt_seq
        
        # Get content seq (piano notes)
        content_seq = get_piano_opd(tgt_seq)

        # Get instrument
        insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        # insts = ['i-0']

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        # if self.split != 'test':
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        # else: # Do not provide target when doing generation for test set
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        #     inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']

        inp_str = ' '.join(inp_seq)
        
        return inp_str


class RetrievalDataset(Dataset):
    '''
    Piano retrieval dataset for reduction
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip().split() for l in data] # a list of strings

        # Filter out data that does not have piano in the target sequence
        if split != 'infer':
            print('Filtering data ...')
            data_new = []
            for sample in data:
                _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(sample)
                # Need to have piano
                if 'i-0' not in tgt_seq and 'i-2' not in tgt_seq:
                    continue
                
                # Piano need to be polyphonic
                ap_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
                ep_opd = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
                t = {'i-0': ap_opd, 'i-2': ep_opd}
                piano_opd = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(t)
                pos_tok = [tok for tok in piano_opd if tok.startswith('o')]
                pitch_tok = [tok for tok in piano_opd if tok.startswith('p')]
                n_pos = len(pos_tok)
                n_pitch = len(pitch_tok)
                if n_pos == n_pitch:
                    continue

                # Filter by piano range
                if 'piano_min_rel_range' in config:
                    piano_min_rel_range = config['piano_min_rel_range']
                    # Get the pitch range of the piano
                    piano_pitch_range = remi_utils.from_remi_get_pitch_range(piano_opd)
                    tgt_pitch_range = remi_utils.from_remi_get_pitch_range(tgt_seq)
                    if tgt_pitch_range == 0:
                        continue
                    rel_range = piano_pitch_range / tgt_pitch_range
                    if rel_range < piano_min_rel_range:
                        continue

                data_new.append(sample)
            self.data = data_new
        else:
            self.data = data

        self.split = split
        self.config = config
        self.rand_inst_infer = rand_inst_infer

        self.remi_aug = ArrangerAugment(config=config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_seq = self.data[index]
       
        # Pitch shift augmentation
        if self.config.get('pitch_shift', False) is True and self.split == 'train':
            remi_seq = remi_utils.in_remi_shift_pitch_random(remi_seq)

        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

        # Get pos and pitch sequence
        tgt_for_content = tgt_seq.copy()
        if self.config.get('content_aug') is True and self.split == 'train':
            tgt_for_content = remi_utils.in_remi_multi_bar_delete_non_mel_insts(tgt_for_content)
        
        if self.config.get('dur_input') is True:
            content_seq = remi_utils.from_remi_get_global_opd_seq(tgt_for_content)
        else:
            content_seq = remi_utils.from_remi_get_pitch_of_pos_seq(tgt_for_content, flatten=True)
        

        def get_piano_opd(tgt_seq):
            # Obtain 'i-0' and 'i-2' tracks, merge together, assign 'i-0' inst
            opd_dict = {}
            if 'i-0' in tgt_seq:
                ap_opd_seq = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-0')
                opd_dict['i-0'] = ap_opd_seq
            if 'i-2' in tgt_seq:
                ep_opd_seq = remi_utils.from_remi_get_opd_seq_of_track(tgt_seq, 'i-2')
                opd_dict['i-2'] = ep_opd_seq
            tgt_seq = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(opd_dict)
            if len(tgt_seq) > 0:
                tgt_seq = ['i-0'] + tgt_seq
            tgt_seq = tgt_seq + ['b-1']

            return tgt_seq
        
        # Get target sequence
        tgt_seq = get_piano_opd(tgt_seq)

        # Modify history sequence
        hist_seq = get_piano_opd(hist_seq)

        # Get instrument
        # insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        insts = ['i-0']

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        # if self.split != 'test':
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        # else: # Do not provide target when doing generation for test set
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        #     inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']

        inp_str = ' '.join(inp_seq)
        
        return inp_str


class DrumArrangeDataset(Dataset):
    '''
    The dataset class that read the 8-bar dataset and prepare the data for 4-bar drum arrangement training
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip().split() for l in data] # a list of strings

        # In training, filter out data that have not enough drum bars in target sequence
        if split != 'infer':
            print('Filtering data ...')
            data_new = []
            for sample in data:
                _, tgt_seq = remi_utils.from_remi_eight_bar_split_hist_tgt_seq(sample)
                drum_bars = tgt_seq.count('i-128')
                if drum_bars >= config['min_drum_bars_in_tgt']:
                    data_new.append(sample)
            self.data = data_new
        else:
            self.data = data

        self.split = split
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_seq = self.data[index]
       
        # # Get time signature
        # ts = remi_seq[0]

        # # Get tempo
        # tempo = remi_seq[1]

        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_eight_bar_split_hist_tgt_seq(remi_seq)

        # Get drum sequence from history
        hist_drum_seq = remi_utils.from_remi_mbar_get_opd_seq_of_track(hist_seq, 'i-128')

        # Split drum and other sequence in the target seq
        tgt_drum_seq = remi_utils.from_remi_mbar_get_opd_seq_of_track(tgt_seq, 'i-128')
        other_inst_seq = remi_utils.from_remi_mbar_remove_drum(tgt_seq)

        if self.config.get('opd_input') is True:
            other_inst_seq = remi_utils.from_remi_mbar_get_global_opd_seq(other_inst_seq)

        ''' Condition format: [BOS] <ts> <tempo> [INST] <inst_seq> [PITCH] <content_seq> [HIST] <hist_seq> [SEP] <tgt_seq> [EOS] '''

        # Assembly the input sequence
        insts = ['i-128']
        # if self.split != 'test':
            # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + other_inst_seq + ['[HIST]'] + hist_drum_seq + ['[SEP]'] + tgt_drum_seq + ['[EOS]']
        # else: # Do not provide target when doing generation for test set
        #     # inp_seq = ['[BOS]', ts, tempo] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        #     inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + other_inst_seq + ['[HIST]'] + hist_drum_seq + ['[SEP]']

        inp_str = ' '.join(inp_seq)
        
        return inp_str
    
    


class ExpansionDataset(ArrangerDataset):
    '''
    This dataset class is used for training the model to generate music from lead sheet.
    The input sequence is constructed by corrupting the target sequence to a lead sheet.

    I.e., 
    - a single instrument track that has the highest average pitch sequence,
    - A chord sequence that is generated from mixture sequence
    
    '''
    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')

        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

        # Get chord sequence from target bar
        chord_seq = remi_utils.from_remi_get_chord_seq(tgt_seq)
        chord_token_seq = remi_utils.from_chord_seq_get_chord_token_seq(chord_seq)
        # Get melody
        melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            tgt_seq,
            monophonic_only=True,
            top_note=False,
        )

        # Get instruments
        insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)

        # Assembly the input sequence
        # if self.split != 'test':
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[MELODY]'] + melody_seq + ['[CHORD]'] + chord_token_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        # else: # Do not provide target when doing generation for test set
        #     inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[MELODY]'] + melody_seq + ['[CHORD]'] + chord_token_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        
        inp_str = ' '.join(inp_seq)

        return inp_str



class ExpansionDatasetChordNoteTwoChords(ArrangerDataset):
    '''
    Represent the lead sheet with remi of full a instrument track
    2 chords per bar
    '''
    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')

        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

        # Get chord sequence from target bar
        chord_seq = remi_utils.from_remi_get_chord_seq_2chord_1bar(tgt_seq)
        chord_opd_dict = remi_utils.from_chord_seq_get_chord_note_opd_dict_2chord_per_bar(chord_seq)

        # Get melody
        _, mel_inst = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            tgt_seq,
            monophonic_only=True,
            top_note=False,
            return_inst=True,
        )
        mel_opd_dict = remi_utils.from_remi_get_opd_dict_of_track(tgt_seq, mel_inst)

        # Merge melody and chord sequence
        lead_sheet_opd_dict = remi_utils.merge_chord_and_melody_opd_dict(chord_opd_dict, mel_opd_dict)
        lead_sheet_seq = remi_utils.dict_to_seq(lead_sheet_opd_dict)

        # Get instruments
        insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + lead_sheet_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        
        inp_str = ' '.join(inp_seq)
        return inp_str


class ExpansionDatasetChordNote(ArrangerDataset):
    '''
    This dataset class is used for training the model to generate music from lead sheet.
    The input sequence is constructed by corrupting the target sequence to a lead sheet.

    I.e., 
    - a single instrument track that has the highest average pitch sequence,
    - A chord sequence that is generated from mixture sequence
    
    '''
    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')
        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)
        # Get chord sequence from target bar
        chord_seq = remi_utils.from_remi_get_chord_seq(tgt_seq)
        chord_note_seq = remi_utils.from_chord_seq_get_chord_note_seq(chord_seq)
        # # Get melody
        # melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
        #     tgt_seq,
        #     monophonic_only=True,
        #     top_note=False,
        # )
        # Get melody
        _, mel_inst = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            tgt_seq,
            monophonic_only=True,
            top_note=False,
            return_inst=True,
        )
        mel_opd_dict = remi_utils.from_remi_get_opd_dict_of_track(tgt_seq, mel_inst)
        mel_seq = remi_utils.dict_to_seq(mel_opd_dict)

        # Get instruments
        insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[MELODY]'] + mel_seq + ['[CHORD]'] + chord_note_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        
        inp_str = ' '.join(inp_seq)
        return inp_str


class ExpanderInferDataset(Dataset):
    '''
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, melody_fp, chord_fp, split, config):
        # Read the remi data (one sample one line, one split one file)
        with open(melody_fp) as f:
            data = f.readlines()
        data = [l.strip().split() for l in data] # a list of strings
        self.melody = data

        with open(chord_fp) as f:
            data = f.readlines()
        data = [l.strip().split() for l in data] # a list of strings
        self.chord = data

        if len(self.chord) > len(self.melody):
            pad_melody = len(self.chord) - len(self.melody)
            self.melody += [['b-1', 'b-1']] * pad_melody

        assert len(self.melody) == len(self.chord)

        self.split = split
        self.config = config

        self.remi_aug = ArrangerAugment(config=config)

    def __len__(self):
        return len(self.melody)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_seq = self.melody[index]
        chords = self.chord[index]

        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)

        # Get chord sequence from target bar
        chord_seq = []
        for chord in chords:
            if chord == 'N':
                chord_seq.append(None)
            else:
                root, quality = chord.split(':')
                chord_seq.append((root, quality))
        chord_note_seq = remi_utils.from_chord_seq_get_chord_note_seq(chord_seq)
        
        # Get melody
        melody_seq = remi_utils.from_remi_get_global_opd_seq(tgt_seq)
        # melody_seq = remi_utils.from_remi_get_pitch_of_pos_seq(tgt_seq, flatten=False)

        # Get instruments
        insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[MELODY]'] + melody_seq + ['[CHORD]'] + chord_note_seq + ['[HIST]'] + hist_seq + ['[SEP]'] + tgt_seq + ['[EOS]']
        
        inp_str = ' '.join(inp_seq)
        return inp_str


class ExpansionDatasetNohist(ArrangerDataset):
    '''
    This dataset class is used for training the model to generate music from lead sheet.
    The input sequence is constructed by corrupting the target sequence to a lead sheet.

    I.e., 
    - a single instrument track that has the highest average pitch sequence,
    - A chord sequence that is generated from mixture sequence
    
    '''
    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')
        # Hist and target sequence
        hist_seq, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(remi_seq)
        # Get chord sequence from target bar
        chord_seq = remi_utils.from_remi_get_chord_seq(tgt_seq)
        chord_note_seq = remi_utils.from_chord_seq_get_chord_note_seq(chord_seq)
        # Get melody
        melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            tgt_seq,
            monophonic_only=True,
            top_note=False,
        )

        # Get instruments
        insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        # Assembly the input sequence
        inp_seq = ['[BOS]'] + ['[INST]'] + insts + ['[MELODY]'] + melody_seq + ['[CHORD]'] + chord_note_seq + ['[HIST]'] + ['[SEP]'] + tgt_seq + ['[EOS]']
        
        inp_str = ' '.join(inp_seq)
        return inp_str



class InstPredDataset(Dataset):
    '''
    This dataset class is used for training the model to predict the instrument label 
    from a given track of a multitrack symbolic music.

    I.e., 
    - a single instrument track that has the highest average pitch sequence,
    - A chord sequence that is generated from mixture sequence
    
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.remi_seqs = [l.split(' ') for l in data]

        # self.indices = {} # map from sample index to (remi_seq id, bar_id, inst type)
        # self.insts_of_bars = {}
        # cnt = 0
        # ''' Segment the remi to bars '''
        # for i, remi_seq in enumerate(self.remi_seqs):
        #     bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
        #     for bar_id in bar_indices:
        #         bar_start_idx, bar_end_idx = bar_indices[bar_id]
        #         bar_seq = self.remi_seqs[0][bar_start_idx:bar_end_idx]
        #         insts_of_bar = remi_utils.from_remi_get_insts(bar_seq)
        #         for inst in insts_of_bar:
        #             self.indices[cnt] = (i, bar_id, inst)
        #             cnt += 1


        ''' Index the data, save the sample index and inst id '''
        self.indices = {} # map from sample index to (remi_seq id, inst type)
        idx = 0
        insts_of_sample = {}
        for i, remi_seq in enumerate(self.remi_seqs):
            insts = remi_utils.from_remi_get_insts(remi_seq)
            insts_of_sample[i] = insts
            for inst in insts:
                self.indices[idx] = (i, inst)
                idx += 1

        self.split = split
        self.config = config
        self.inst_util = InstMapUtil()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ''' Get the sample index and inst type '''
        remi_seq_id, inst = self.indices[index]
        
        remi_seq = self.remi_seqs[remi_seq_id]

        program_id = int(inst.split('-')[1])

        tot_note_seq = []
        bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_seq = remi_seq[bar_start_idx:bar_end_idx]
            notes_of_track = remi_utils.from_remi_get_opd_seq_of_track(bar_seq, inst)
            tot_note_seq.extend(notes_of_track)
            tot_note_seq.append('b-1')

        return tot_note_seq, program_id


class InstPredDatasetPaired(Dataset):
    '''
    This dataset class is used for training the model to predict the instrument label 
    from a given track of a multitrack symbolic music.

    Return track wise remi for binary classification

    I.e., 
    - a single instrument track that has the highest average pitch sequence,
    - A chord sequence that is generated from mixture sequence
    
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.remi_seqs = [l.split(' ') for l in data]

        ''' Index the data, save the sample index and inst id '''
        self.indices = {} # map from sample index to (remi_seq id, inst type)
        idx = 0
        insts_of_sample = {}
        for i, remi_seq in enumerate(self.remi_seqs):
            insts = remi_utils.from_remi_get_insts(remi_seq)
            insts_of_sample[i] = insts
            for inst in insts:
                self.indices[idx] = (i, inst)
                idx += 1
        self.insts_of_sample = insts_of_sample

        self.split = split
        self.config = config
        self.inst_util = InstMapUtil()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ''' Get the sample index and inst type '''
        remi_seq_id, inst = self.indices[index]
        
        insts_of_sample = self.insts_of_sample[remi_seq_id]

        # 50% chance to use correct inst, 50% chance to use incorrect inst
        if random.random() < 0.5:
            # Use correct inst combination
            label = 1
        else:
            # User incorrect inst combination
            # Remove original inst from insts_of_sample
            insts = insts_of_sample.copy()
            insts.remove(inst)
            if len(insts) == 0:
                insts.append('i-128')
            new_inst = random.choice(insts)
            inst = new_inst
            label = 0

        remi_seq = self.remi_seqs[remi_seq_id]


        tot_note_seq = []
        bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_seq = remi_seq[bar_start_idx:bar_end_idx]
            notes_of_track = remi_utils.from_remi_get_opd_seq_of_track(bar_seq, inst)
            tot_note_seq.extend(inst)
            tot_note_seq.extend(notes_of_track)
            tot_note_seq.append('b-1')

        return tot_note_seq, label


class ChordPredDataset(Dataset):
    '''
    This dataset class is used for training the model to predict the chord label
    from a multi-track symbolic music.
    '''
    def __init__(self, data_fp, split, config, rand_inst_infer=False):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings

        # ''' Process the data, extract each track as a sample '''
        # track_data = []
        # pbar = tqdm(data)
        # for line in pbar:
        #     pbar.set_description('Loading {} set ...'.format(split))
        #     sample_seq = line.split(' ')
        #     t = remi_utils.from_remi_get_pos_and_pitch_seq_per_track(sample_seq)
        #     track_data.extend([(t[inst], inst) for inst in t if inst != 'i-128'])

        # self.data = track_data
        self.data = data
        self.split = split
        self.config = config
        self.inst_util = InstMapUtil()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        remi_str = self.data[index]
        remi_seq = remi_str.strip().split(' ')

        # Get the note of all insts
        new_remi = []
        b_1_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
        for bar_id in b_1_indices:
            bar_start_idx, bar_end_idx = b_1_indices[bar_id]
            bar_seq = remi_seq[bar_start_idx:bar_end_idx]
            
            opd_seq_of_track = remi_utils.from_remi_get_opd_seq_per_track(bar_seq, sort_by_avg_pitch=True)
            
            # Remove drum
            if 'i-128' in opd_seq_of_track:
                opd_seq_of_track.pop('i-128')
            
            # Merge all notes into a single sequence, sort by position
            # Within position, sort by pitch, descending
            merged_seq = remi_utils.from_remi_reordered_opd_dict_merge_to_single_sequence(opd_seq_of_track)
            new_remi.extend(merged_seq)
            new_remi.append('b-1')

        remi_seq = new_remi

        # Get first b-1 token's position
        b_1_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
        bar_1 = remi_seq[b_1_indices[0][0]:b_1_indices[0][1]]
        bar_2 = remi_seq[b_1_indices[1][0]:b_1_indices[1][1]]

        # Get the chord labels
        chord_seq_1 = remi_utils.from_remi_get_chord_seq_two_chord_a_bar(bar_1)
        chord_seq_2 = remi_utils.from_remi_get_chord_seq_two_chord_a_bar(bar_2)

        # Convert chord label to root id and type id
        chord_id_1 = [chord_to_id(c) for c in chord_seq_1]
        chord_id_2 = [chord_to_id(c) for c in chord_seq_2]
        labels = chord_id_1 + chord_id_2

        root_labels = [l[0] for l in labels]
        type_labels = [l[1] for l in labels]

        return remi_seq, root_labels, type_labels


class ArrangerAugment:
    '''
    This class define several modification operations to the remi sequence "<condition> <sep> <target>".
    Contains 5 different tasks, and 2 additional augmentation operations.
    '''

    def __init__(self, config) -> None:
        self.tasks = [
            # self.task1_reconstruction,
            # self.task2_content_simplification,
            # self.task3_content_elaboration
            self.task2_arrangement,
        ]
        self.pitch_reorder = False
        self.pitch_shift = False

        self.config = config

        # self.hist = config['with_hist']
        # self.voice_control = config['voice_control']
        # self.texture_control = config['texture_control']
        # self.flatten_content = config['flatten_content']
        # self.aug_hist = config['aug_hist'] if 'aug_hist' in config else False
        

    def select_and_apply_task(self, condition_seq, remi_seq):
        '''
        Random select one of the task from self.tasks
        Apply corresponding task to the input and output sequence
        Insert the task token to the beginning of condition sequence
        '''
        # Modify input and output according one specific task
        task = random.choice(self.tasks)
        condition_seq, remi_seq = task(condition_seq, remi_seq)

        return condition_seq, remi_seq

    def aug_inst_del_insts_from_tgt(self, condition_seq, tgt_remi_seq, insts_to_del):
        # Track deletion, in target sequence (changes will reflect in condition seq by construct it again)
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq)
        num_insts = len(insts)
        if num_insts <= 1:
            return condition_seq, tgt_remi_seq

        new_insts = insts.copy()
        for inst_to_del in insts_to_del:
            if inst_to_del in insts:
                new_insts.remove(inst_to_del)

        # Modify the remi, delete notes do not needed.
        tgt_remi_seq = self.__retain_specified_insts_in_remi(tgt_remi_seq, new_insts) # NOTE: might be buggy for 2-bar segment
        new_insts = remi_utils.in_inst_list_sort_inst(list(new_insts))

        # Modify inst in conditions according to track deletion augmentation
        condition_seq = remi_utils.in_condition_keep_only_specified_insts(
            condition_seq, 
            new_insts,
            has_texture=self.texture_control
        )

        return condition_seq, tgt_remi_seq

    def augment_remi(self, condition_seq, tgt_remi_seq):
        '''
        Conduct the task selection and augmentation
        '''
        # For debugging
        if len(tgt_remi_seq) > 2:
            a = 1
        
        # Augmentation 1: instrument aug
        t = random.uniform(0, 1)
        if t > 0.6666:
            condition_seq, tgt_remi_seq = self.aug_inst_pred_single_inst(condition_seq, tgt_remi_seq)
        elif t > 0.3333:
            condition_seq, tgt_remi_seq = self.aug_inst_del_one(condition_seq, tgt_remi_seq)
        else:
            pass # (1/3 chance input content same as output)

        # Augmentation 2: drum deletion from target and inst spec
        t = random.uniform(0, 1)
        if t > 0.6666:
            condition_seq, tgt_remi_seq = self.aug_inst_del_drum(condition_seq, tgt_remi_seq)
        elif t > 0.3333:
            condition_seq, tgt_remi_seq = self.aug_inst_pred_drum(condition_seq, tgt_remi_seq)
        else:
            pass

        # Augmentation 3: bass deletion from target and inst spec
        t = random.uniform(0, 1)
        if t > 0.6666:
            condition_seq, tgt_remi_seq = self.aug_inst_del_bass(condition_seq, tgt_remi_seq)
        elif t > 0.3333:
            condition_seq, tgt_remi_seq = self.aug_inst_pred_bass(condition_seq, tgt_remi_seq)
        else:
            pass

        # Augmentation 4: delete drum seq from history
        t = random.uniform(0, 1)
        if t > 0.5:
            condition_seq, tgt_remi_seq = self.aug_hist_del_drum(condition_seq, tgt_remi_seq)

        # Augmentation 5: random history deletion
        if self.aug_hist:
            t = random.uniform(0, 1)
            if t > 0.9:
                condition_seq, tgt_remi_seq = self.aug_hist_random_del(condition_seq, tgt_remi_seq)

        # TODO: pitch reorder # May not be necessary ... ?

        # TODO: pitch shift

        return condition_seq, tgt_remi_seq
    
    def aug_hist_random_del(self, condition_seq, tgt_remi_seq):
        '''
        Delete entire history from condition_seq (but keep the HIST token)
        '''
        # print('Del hist aug!')
        if 'HIST' not in condition_seq:
            return condition_seq, tgt_remi_seq
        
        hist_idx = condition_seq.index('HIST')
        if hist_idx == len(condition_seq) - 1:
            return condition_seq, tgt_remi_seq
        
        hist_start_idx = hist_idx + 1
        condition_seq = condition_seq[:hist_start_idx]

        return condition_seq, tgt_remi_seq

    def aug_hist_del_drum(self, condition_seq, tgt_remi_seq):
        '''
        Delete drum history from condition seq, keep tgt as-is
        '''
        # print('del_drum_hist')
        if 'HIST' not in condition_seq:
            return condition_seq, tgt_remi_seq

        hist_idx = condition_seq.index('HIST')
        if hist_idx == len(condition_seq) - 1:
            return condition_seq, tgt_remi_seq

        hist_start_idx = hist_idx + 1
        hist_end_idx = len(condition_seq)
        ori_hist = condition_seq[hist_start_idx:hist_end_idx]

        if 'i-128' not in ori_hist:
            return condition_seq, tgt_remi_seq

        # Filter out drum history
        drum_hist_idx = ori_hist.index('i-128')
        new_hist = ori_hist[:drum_hist_idx]

        # Reconstruct the condition sequence
        new_condition_seq = condition_seq[:hist_start_idx] + new_hist

        return new_condition_seq, tgt_remi_seq


    def task1_reconstruction(self, condition_seq, remi_seq):
        # Append task tokens (to the end of condition sequence)
        task_tokens = 'X-0'
        condition_seq.insert(0, task_tokens)

        return condition_seq, remi_seq
    
    def task2_arrangement(self, condition_seq, tgt_remi_seq):
        # print('arrange is selected')
        # Augmentation for arrangement task

        # Augmentation 1: instrument aug
        if self.config.get('aug_inst_slight') is True:
            t = random.uniform(0, 1)
            if t > 0.6666:
                condition_seq, tgt_remi_seq = self.aug_inst_pred_single_inst(condition_seq, tgt_remi_seq)
            elif t > 0.3333:
                condition_seq, tgt_remi_seq = self.aug_inst_del_one(condition_seq, tgt_remi_seq)
            else:
                pass # (1/3 chance input content same as output)

        # Augmentation 2: drum deletion from target and inst spec, as well as history
        if self.config.get('aug_drum') is True:
            t = random.uniform(0, 1)
            if t > 0.6666:
                condition_seq, tgt_remi_seq = self.aug_inst_del_drum(condition_seq, tgt_remi_seq)
            elif t > 0.3333:
                condition_seq, tgt_remi_seq = self.aug_inst_pred_drum(condition_seq, tgt_remi_seq)
            else:
                pass

            # delete drum seq from history
            t = random.uniform(0, 1)
            if t > 0.5:
                condition_seq, tgt_remi_seq = self.aug_hist_del_drum(condition_seq, tgt_remi_seq)

        # Augmentation 3: Bass augmentation
        if self.config.get('aug_bass') is True:
            # Augmentation 3: bass deletion from target and inst spec
            t = random.uniform(0, 1)
            if t > 0.6666:
                condition_seq, tgt_remi_seq = self.aug_inst_del_bass(condition_seq, tgt_remi_seq)
            elif t > 0.3333:
                condition_seq, tgt_remi_seq = self.aug_inst_pred_bass(condition_seq, tgt_remi_seq)
            else:
                pass

        # Augmentation 4: random history deletion
        if self.config.get('aug_hist') is True:
            t = random.uniform(0, 1)
            if t > 0.9:
                condition_seq, tgt_remi_seq = self.aug_hist_random_del(condition_seq, tgt_remi_seq)

        # Augmentation 5: severe inst aug: infilling and retrieval
        if self.config.get('aug_inst') is True:
            t = random.uniform(0, 1)
            if t > 0.6666:
                condition_seq, tgt_remi_seq = self.aug_inst_tracks_infill(condition_seq, tgt_remi_seq)
            elif t > 0.3333:
                condition_seq, tgt_remi_seq = self.aug_inst_tracks_del(condition_seq, tgt_remi_seq)

        # Augmentation 6: denoise (random del for each track)
        if self.config.get('aug_denoise') is True:
            t = random.uniform(0, 1)
            if t > 0.5:
                condition_seq, tgt_remi_seq = self.aug_inst_tracks_denoise(condition_seq, tgt_remi_seq)

        # Augmentation 7: random content deletion / additive noise
        if self.config.get('aug_content') is True:
            t = random.uniform(0, 1)
            if t > 0.6666:
                condition_seq, tgt_remi_seq = self.aug_content_random_deletion(condition_seq, tgt_remi_seq)
            elif t > 0.3333:
                condition_seq, tgt_remi_seq = self.aug_content_additive_noise(condition_seq, tgt_remi_seq)

        

        # TODO: pitch reorder # May not be necessary ... ?

        # TODO: pitch shift

        # # Append task tokens (to the end of condition sequence)
        # task_tokens = 'X-1'
        # condition_seq.insert(0, task_tokens)

        return condition_seq, tgt_remi_seq

    def aug_content_additive_noise(self, condition_seq: List[str], remi_seq: List[str]):
        """Adjust the input pitch sequence, adding more pitch and position tokens inside,
        So that the output content is simpler than input
        
        On average, 25% additional position tokens are added, 
            each with additional (avg_n_pitch/pos) pitch tokens per position
        Additionaly, 25% pitch tokens are randomly added to all positions

        NOTE: If condition is empty (not a single pitch), no modification will be made. 
        
        Args:
            condition_seq (List[str]): condition sequence of a segment
            remi_seq (List[str]): remi sequence of the segment

        Returns:
            Modified condition_seq and original remi_seq
        """        
        new_segment_condition_seq = []

        bar_condition_seq = condition_seq

        # Calculate the average pitch per position for the sample
        hist_pos = condition_seq.index('HIST') if 'HIST' in condition_seq else len(condition_seq)
        pos_cnt, pitch_cnt = remi_utils.from_condition_get_pos_and_pitch(condition_seq[:hist_pos])
        if pitch_cnt == 0: # If it's an empty bar, do not change condition
            new_segment_condition_seq.extend(bar_condition_seq)
        else: # If bar not empty
            ''' Randomly add more pitch tokens in content seq '''
            # Achieved by random adding pitch tokens to pitch seq in condition 

            # Get the pitch of each position
            # Note: current condition: pitch, inst, hist
            pitch_seq_start_idx = bar_condition_seq.index('PITCH')
            pitch_seq_end_idx = bar_condition_seq.index('INS')
            pos_and_pitch_seq = bar_condition_seq[pitch_seq_start_idx+1:pitch_seq_end_idx]
            pitch_of_pos = remi_utils.from_pitch_of_pos_seq_get_pitch_of_pos_dict(pos_and_pitch_seq)

            # Find highest pitch of each pos for pitch id <= 127
            highest_pitch_of_pos = {}
            for pos in pitch_of_pos:
                non_drum_pitch = [remi_utils.from_pitch_token_get_pitch_id(p) for p in pitch_of_pos[pos] if remi_utils.from_pitch_token_get_pitch_id(p) <= 127]
                if len(non_drum_pitch) == 0:
                    highest_pitch_of_pos[pos] = 127
                else:
                    highest_pitch_of_pos[pos] = max(non_drum_pitch)

            # Determine the number of pitch to add
            avg_num_new_pitch = max(1, pitch_cnt // 4) # Random add 25% content to each position
            num_new_pitch = max(1, np.random.poisson(avg_num_new_pitch))
            
            # For each additional token
            for i in range(num_new_pitch):
                # Random select a position to add pitch
                pos = random.choice(list(highest_pitch_of_pos.keys()))

                # Get a random pitch token
                new_p_tok = self.__get_random_pitch_token(upper=highest_pitch_of_pos[pos])

                # Add it to pitch_of_pos
                pitch_of_pos[pos].append(new_p_tok)

            # Reconstruct the pitch sequence
            new_pitch_seq = []
            for pos in pitch_of_pos:
                # Sort pitch tokens by pitch id
                pitch_of_pos[pos].sort(key=lambda x: remi_utils.from_pitch_token_get_pitch_id(x), reverse=True)
                new_pitch_seq.append(pos)
                new_pitch_seq.extend(pitch_of_pos[pos])

            # Insert the new pitch sequence to the original condition sequence
            new_bar_condition_seq = bar_condition_seq[:pitch_seq_start_idx+1] + new_pitch_seq + bar_condition_seq[pitch_seq_end_idx:]

            new_segment_condition_seq.extend(new_bar_condition_seq)

        # # Append task tokens (to the end of condition sequence)
        # task_tokens = 'X-2'
        # new_segment_condition_seq.insert(0, task_tokens)

        return new_segment_condition_seq, remi_seq
    

    def old_task2_content_simplification(self, condition_seq: List[str], remi_seq: List[str]):
        """Adjust the input pitch sequence, adding more pitch and position tokens inside,
        So that the output content is simpler than input
        
        On average, 25% additional position tokens are added, 
            each with additional (avg_n_pitch/pos) pitch tokens per position
        Additionaly, 25% pitch tokens are randomly added to all positions

        NOTE: If condition is empty (not a single pitch), no modification will be made. 
        
        Args:
            condition_seq (List[str]): condition sequence of a segment
            remi_seq (List[str]): remi sequence of the segment

        Returns:
            Modified condition_seq and original remi_seq
        """        
        new_segment_condition_seq = []

        bar_condition_seq = condition_seq

        ''' Randomly add some position tokens: more complex rhythm '''
        # Achieved by introduce additional position tokens with a sequence of random pitch tokens.
        # Calculate the average pitch per position for the sample
        hist_pos = condition_seq.index('HIST')
        pos_cnt, pitch_cnt = remi_utils.from_condition_get_pos_and_pitch(condition_seq[:hist_pos])
        if pitch_cnt == 0: # If it's an empty bar, do not change condition
            new_segment_condition_seq.extend(bar_condition_seq)
        else: # If bar not empty
            # avg_p_per_o = int(pitch_cnt / pos_cnt) # should a value >= 1
            # # Determine the number new positions: 
            # avg_num_new_pos = max(1, pos_cnt // 4) # We expect 25% more positions added to the content
            # num_new_pos = max(1, np.random.poisson(avg_num_new_pos))

            # # For each new position,       5-21: do not modify pos when doing aug
            # for i in range(num_new_pos):
            #     # Determine the locations of the new positions: random choice
            #     new_pos_tok = self.__get_random_position_token()

            #     # Determine the number of pitch tokens for the new position
            #     num_pitch_token = max(1, np.random.poisson(lam=avg_p_per_o))

            #     # Prepare a subsequence of (o-X p-Y p-Z ...)
            #     p_subseq = self.__get_random_pitch_tokens(n_tok=num_pitch_token)
            #     subseq = [new_pos_tok] + p_subseq

            #     # Insert the subsequence to the proper place in the input sequence
            #     bar_condition_seq = self.__insert_subseq_to_condition_for_a_bar(bar_condition_seq, subseq)

            ''' Randomly add more pitch tokens: more complex harmony '''
            # Achieved by random adding pitch tokens to pitch seq in condition 
            # Determine the number of pitch to add
            avg_num_new_pitch = max(1, pitch_cnt // 4) # Random add 25% content to each position
            num_new_pitch = max(1, np.random.poisson(avg_num_new_pitch))
            
            # Obtain the location of the pitch sequence
            pitch_tok_idx = bar_condition_seq.index('PITCH')

            # Note: current condition: pitch, inst, hist
            # For each additional token
            for i in range(num_new_pitch):
                # Random select a location in input
                inst_tok_pos = bar_condition_seq.index('INS')
                idx = random.randint(pitch_tok_idx+1, inst_tok_pos)
                # Insert it to the input sequence
                new_p_tok = self.__get_random_pitch_token()
                bar_condition_seq.insert(idx, new_p_tok)

            new_segment_condition_seq.extend(bar_condition_seq)

        # Append task tokens (to the end of condition sequence)
        task_tokens = 'X-2'
        new_segment_condition_seq.extend(task_tokens)

        return new_segment_condition_seq, remi_seq
    

    def aug_content_random_deletion(self, condition_seq, remi_seq):
        '''
        Adjust the input pitch sequence, delete some random content from pitch sequence
        So that the output content is more complex than input
        Keep in instrument prompt as-is. Keep the target sequence as-is.
        '''
        new_segment_condition_seq = []

        bar_condition_seq = condition_seq
        hist_pos = bar_condition_seq.index('HIST') if 'HIST' in bar_condition_seq else len(bar_condition_seq)
        _, pitch_cnt = remi_utils.from_condition_get_pos_and_pitch(bar_condition_seq[:hist_pos])

        ''' Randomly delete pitch and position tokens '''
        content_start_idx = bar_condition_seq.index('PITCH') + 1
        content_end_idx = bar_condition_seq.index('INS')

        if content_end_idx <= content_start_idx: # for empty bar, don't do anything
            new_segment_condition_seq.extend(bar_condition_seq)
        else: # If not empty, do the random deletion.
            # 截取指定索引范围的部分
            content_segment = bar_condition_seq[content_start_idx:content_end_idx]
            
            # 计算需要删除的元素数量
            avg_num_to_remove = pitch_cnt // 4
            num_to_remove = np.random.poisson(lam=avg_num_to_remove)
            num_to_remove = max(1, num_to_remove)
            num_to_remove = min(content_end_idx - content_start_idx, num_to_remove)
            
            # Get pitch of each position
            pitch_of_pos = remi_utils.from_pitch_of_pos_seq_get_pitch_of_pos_dict(content_segment)

            # Drop empty positions
            pitch_of_pos = {pos: pitch_of_pos[pos] for pos in pitch_of_pos if len(pitch_of_pos[pos]) > 0}
            if len(pitch_of_pos) == 0:
                new_segment_condition_seq.extend(bar_condition_seq)
            
            else:
                # Get the largest pitch of each position (<=127)
                highest_pitch_of_pos = {}
                for pos in pitch_of_pos:
                    non_drum_pitch = [remi_utils.from_pitch_token_get_pitch_id(p) for p in pitch_of_pos[pos] if remi_utils.from_pitch_token_get_pitch_id(p) <= 127]
                    if len(non_drum_pitch) == 0:
                        highest_pitch_of_pos[pos] = 127 # Drum note can be deleted
                    else:
                        highest_pitch_of_pos[pos] = max(non_drum_pitch)

                # Delete some pitch tokens
                for i in range(num_to_remove):
                    # If empty, break
                    if len(pitch_of_pos) == 0:
                        break

                    # Get a random pos to delete pitch
                    pos = random.choice(list(pitch_of_pos.keys()))
                    
                    # Random delete a pitch token that is lower than the highest pitch of the pos
                    # from the pitch seq of the pos
                    pitch_seq = pitch_of_pos[pos]

                    # Get the highest pitch of the pos
                    highest_pitch = highest_pitch_of_pos[pos]

                    # Find the indices of pitch tokens that are lower than the highest pitch
                    lower_pitch_indices = [i for i, pitch in enumerate(pitch_seq) if remi_utils.from_pitch_token_get_pitch_id(pitch) < highest_pitch or remi_utils.from_pitch_token_get_pitch_id(pitch) >= 128]

                    # Randomly select an index from the lower_pitch_indices
                    if lower_pitch_indices:
                        idx = random.choice(lower_pitch_indices)

                        # Delete the selected pitch token
                        pitch_seq.pop(idx)

                    # If the pitch seq of the pos is empty, remove the pos
                    if len(pitch_seq) == 0:
                        del pitch_of_pos[pos]

                # Reconstruct the content segment
                new_content_segment = []
                for pos in pitch_of_pos:
                    new_content_segment.append(pos)
                    new_content_segment.extend(pitch_of_pos[pos])

                # 重建整个列表，保持其他部分不变
                new_bar_condition_seq = bar_condition_seq[:content_start_idx] + new_content_segment + bar_condition_seq[content_end_idx:]
                new_segment_condition_seq.extend(new_bar_condition_seq)

        # # Insert task tokens (to the beginning of condition sequence)
        # task_tokens = 'X-1'
        # new_segment_condition_seq.insert(0, task_tokens)

        return new_segment_condition_seq, remi_seq
    

    def bak_task3_content_elaboration(self, condition_seq, remi_seq):
        '''
        Adjust the input pitch sequence, delete some random content from pitch sequence
        So that the output content is more complex than input
        Keep in instrument prompt as-is. Keep the target sequence as-is.
        '''
        new_segment_condition_seq = []

        bar_condition_seq = condition_seq
        hist_pos = bar_condition_seq.index('HIST')
        _, pitch_cnt = remi_utils.from_condition_get_pos_and_pitch(bar_condition_seq[:hist_pos])

        ''' Randomly delete pitch and position tokens '''
        content_start_idx = bar_condition_seq.index('PITCH') + 1
        content_end_idx = bar_condition_seq.index('INS')

        if content_end_idx <= content_start_idx: # for empty bar, don't do anything
            new_segment_condition_seq.extend(bar_condition_seq)
        else: # If not empty, do the random deletion.
            # 截取指定索引范围的部分
            content_segment = bar_condition_seq[content_start_idx:content_end_idx]
            
            # 计算需要删除的元素数量
            avg_num_to_remove = pitch_cnt // 4
            num_to_remove = np.random.poisson(lam=avg_num_to_remove)
            num_to_remove = max(1, num_to_remove)
            num_to_remove = min(content_end_idx - content_start_idx, num_to_remove)
            
            # 随机选择要删除的元素索引
            p_indices = [index for index, item in enumerate(content_segment) if item.startswith('p-')]
            num_to_remove = min(len(p_indices)-1, num_to_remove)
            indices_to_remove = random.sample(p_indices, num_to_remove)
            
            # 删除选中的元素
            content_segment = [item for idx, item in enumerate(content_segment) if idx not in indices_to_remove]
            
            # Filter out empty positions
            filtered_list = []
            i = 0
            while i < len(content_segment):
                if content_segment[i].startswith('o-'):
                    # Check if this 'o-' is followed by a 'p-'
                    if i + 1 < len(content_segment) and content_segment[i + 1].startswith('p-'):
                        filtered_list.append(content_segment[i])
                elif content_segment[i].startswith('p-'):
                    filtered_list.append(content_segment[i])
                i += 1
            content_segment = filtered_list

            # 重建整个列表，保持其他部分不变
            new_bar_condition_seq = bar_condition_seq[:content_start_idx] + content_segment + bar_condition_seq[content_end_idx:]
            new_segment_condition_seq.extend(new_bar_condition_seq)

        # Insert task tokens (to the beginning of condition sequence)
        task_tokens = 'X-1'
        new_segment_condition_seq.insert(0, task_tokens)

        return new_segment_condition_seq, remi_seq
    

    def aug_inst_pred_insts(self, condition_seq, tgt_remi_seq, insts_to_pred, keep_melody=True):
        '''
        Delete multiple insts from condition seq, keep tgt as-is
        '''
        # Save the original instrument spec
        inst_spec = remi_utils.from_condition_get_inst_spec(condition_seq)

        # Obtain the instruments in the target sequence (results sorted)
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq, sort_inst=True)

        # When empty or only one instrument, return directly
        if len(insts) <= 1:
            return condition_seq, tgt_remi_seq

        ''' Melody Keeping '''
        if keep_melody is True:
            mel_inst = self.__get_melody_inst(tgt_remi_seq, insts)
            if mel_inst in insts_to_pred:
                insts_to_pred.remove(mel_inst)

        # Remove specified instruments
        tgt_seq_for_condition = self.__remove_specified_insts_in_remi(tgt_remi_seq, insts_to_pred)

        new_condition_seq, _ = remi_utils.from_remi_get_condition_seq(
            tgt_seq_for_condition,
            hist=False,
            voice=self.voice_control,
            texture=self.texture_control,
            flatten_content=self.flatten_content,
        )

        # Recover the history part
        hist_start_idx = condition_seq.index('HIST') if 'HIST' in condition_seq else len(condition_seq)
        hist_end_idx = len(condition_seq)
        new_condition_seq.extend(condition_seq[hist_start_idx:hist_end_idx])

        # Recover the instrument prompt
        new_condition_seq = remi_utils.in_condition_seq_replace_inst(new_condition_seq, inst_spec)

        return new_condition_seq, tgt_remi_seq
    
    def aug_inst_pred_single_inst(self, condition_seq, tgt_remi_seq):
        '''
        Adjust condition so that the target has one more instrument than the input
        Delete the content from a certain instrument from input content
        '''
        # Save the original instrument spec
        inst_spec = remi_utils.from_condition_get_inst_spec(condition_seq)

        # Obtain the instruments in the target sequence (results sorted)
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq, sort_inst=True)

        # When empty or only one instrument, return directly
        if len(insts) <= 1:
            return condition_seq, tgt_remi_seq

        ''' Melody Keeping '''
        mel_inst = self.__get_melody_inst(tgt_remi_seq, insts)

        # Delete melody instrument from insts, save to non_mel_insts
        non_mel_insts = insts.copy()
        non_mel_insts.remove(mel_inst)

        # Select an instrument to predict. NOTE: Never select melody instrument to predict
        inst_to_pred = random.choice(non_mel_insts)

        # Remove the instrument in target sequence, re-generate the condition seq
        tgt_seq_for_condition = self.__remove_specified_insts_in_remi(tgt_remi_seq, [inst_to_pred])
        new_condition_seq, _ = remi_utils.from_remi_get_condition_seq(
            tgt_seq_for_condition,
            hist=False,
            voice=self.voice_control,
            texture=self.texture_control,
            flatten_content=self.flatten_content,
        )

        # Recover the history part
        hist_start_idx = condition_seq.index('HIST') if 'HIST' in condition_seq else len(condition_seq)
        hist_end_idx = len(condition_seq)
        new_condition_seq.extend(condition_seq[hist_start_idx:hist_end_idx])

        # Recover the instrument prompt
        new_condition_seq = remi_utils.in_condition_seq_replace_inst(new_condition_seq, inst_spec)

        return new_condition_seq, tgt_remi_seq


    def aug_inst_tracks_del(self, condition_seq, tgt_remi_seq):
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq)
        if len(insts) <= 1:
            return condition_seq, tgt_remi_seq
        
        # Melody preservation
        mel_inst = self.__get_melody_inst(tgt_remi_seq, insts)
        non_mel_insts = insts.copy()
        non_mel_insts.remove(mel_inst)

        # Determine the number of instruments to delete
        lamb = len(non_mel_insts) // 2
        num_inst_del = np.random.poisson(lamb)
        num_inst_del = min(num_inst_del, len(non_mel_insts))
        num_inst_del = max(num_inst_del, 1)

        # Randomly select instruments to delete
        insts_to_del = random.sample(non_mel_insts, num_inst_del)

        condition_seq, tgt_remi_seq = self.aug_inst_del_insts_from_tgt(
            condition_seq, 
            tgt_remi_seq, 
            insts_to_del
        )

        return condition_seq, tgt_remi_seq
        

    def __get_melody_inst(self, remi_seq, insts):
        
        # Get the track of each instrument
        track_of_inst = remi_utils.from_remi_get_pitch_seq_per_track(remi_seq)
        
        # Remove drum for now
        if 'i-128' in track_of_inst:
            del track_of_inst['i-128']
        
        # Compute the average pitch id for all instruments
        avg_pitch_id = {}
        for inst in track_of_inst:
            pitch_seq = track_of_inst[inst]
            avg_pitch_id[inst] = np.mean([remi_utils.from_pitch_token_get_pitch_id(p) for p in pitch_seq])
        
        # Get the instrument with the highest average pitch id
        mel_inst = max(avg_pitch_id, key=avg_pitch_id.get)

        return mel_inst

    def aug_inst_tracks_denoise(self, condition_seq, tgt_remi_seq):
        '''
        Reconstruct the content so that on average
        50% of the notes are deleted from each non-melody track,
        the rest are kept
        '''
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq)
        if len(insts) <= 1:
            return condition_seq, tgt_remi_seq

        ''' Melody Keeping '''
        mel_inst = self.__get_melody_inst(tgt_remi_seq, insts)
        non_mel_insts = insts.copy()
        non_mel_insts.remove(mel_inst)

        pos_and_pitch_dict_tot = {}

        # Random deletion of notes in non-melody tracks
        for inst in non_mel_insts:
            pitch_of_pos_dict = remi_utils.from_remi_get_pos_and_pitch_dict_of_track(tgt_remi_seq, inst)
            for pos in pitch_of_pos_dict:
                note_seq = pitch_of_pos_dict[pos]

                # Determine the number of notes to delete
                lamb = len(note_seq) // 2
                num_notes_del = np.random.poisson(lamb)
                num_notes_del = min(num_notes_del, len(note_seq)-1)
                num_notes_del = max(num_notes_del, 0)

                # Randomly select notes to delete
                notes_to_del = random.sample(note_seq, num_notes_del)

                # Remove the selected notes
                new_note_seq = [note for note in note_seq if note not in notes_to_del]
                pitch_of_pos_dict[pos] = new_note_seq

                if len(new_note_seq) > 0:
                    if pos not in pos_and_pitch_dict_tot:
                        pos_and_pitch_dict_tot[pos] = []
                    pos_and_pitch_dict_tot[pos].extend(new_note_seq)

        # Add melody track
        pitch_of_pos_mel = remi_utils.from_remi_get_pos_and_pitch_dict_of_track(tgt_remi_seq, mel_inst)
        for pos in pitch_of_pos_mel:
            if pos not in pos_and_pitch_dict_tot:
                pos_and_pitch_dict_tot[pos] = []
            pos_and_pitch_dict_tot[pos].extend(pitch_of_pos_mel[pos])

        pos_and_pitch_seq = []
        all_pos = pos_and_pitch_dict_tot.keys()
        # Sort the positions
        all_pos = sorted(all_pos, key=lambda x: int(x.split('-')[1]))
        for pos in all_pos:
            pos_and_pitch_seq.append(pos)
            # sort the pitch tokens by pitch id
            pos_and_pitch_dict_tot[pos].sort(key=lambda x: remi_utils.from_pitch_token_get_pitch_id(x), reverse=True)
            pos_and_pitch_seq.extend(pos_and_pitch_dict_tot[pos])

        content_start_idx = condition_seq.index('PITCH') + 1
        content_end_idx = condition_seq.index('INS')
        new_condition_seq = condition_seq[:content_start_idx] + pos_and_pitch_seq + condition_seq[content_end_idx:]

        return new_condition_seq, tgt_remi_seq


    def aug_inst_del_one(self, condition_seq, tgt_remi_seq):
        '''
        Adjust the target sequence, delete one instrument from target
        Adjust the instrument prompt, delete corresponding instrument
        '''
        # Track deletion, in target sequence (changes will reflect in condition seq by construct it again)
        inst = remi_utils.from_remi_get_insts(tgt_remi_seq)
        num_insts = len(inst)
        if num_insts <= 1:
            return condition_seq, tgt_remi_seq

        # Determine new number of instruments
        num_inst_del = 1
        num_inst_del = min(num_inst_del, num_insts - 1) # delete num_insts - 1 instruments at most
        num_inst_del = max(num_inst_del, 1) # delete 1 instrument at least
        num_inst_new = num_insts - num_inst_del

        ''' Retain Melody '''
        # Get non-melody instruments
        mel_inst = self.__get_melody_inst(tgt_remi_seq, inst)
        non_mel_insts = inst.copy()
        non_mel_insts.remove(mel_inst)

        # Determine the instrument to retain, NOTE: always keep melody instrument
        new_inst = random.sample(non_mel_insts, num_inst_new - 1)
        new_inst.append(mel_inst)

        # Modify the remi, delete notes do not needed.
        tgt_remi_seq = self.__retain_specified_insts_in_remi(tgt_remi_seq, new_inst) # NOTE: might be buggy for 2-bar segment
        new_inst = remi_utils.in_inst_list_sort_inst(list(new_inst))

        # Modify inst in conditions according to track deletion augmentation
        condition_seq = remi_utils.in_condition_keep_only_specified_insts(
            condition_seq, 
            new_inst,
            has_texture=self.texture_control
        )

        return condition_seq, tgt_remi_seq

    def aug_inst_pred_drum(self, condition_seq, tgt_remi_seq):
        '''
        Delete drum from condition seq, keep tgt as-is
        '''
        # print('pred_drum')
        insts_to_pred = ['i-128']
        condition_seq, tgt_remi_seq = self.aug_inst_pred_insts(
            condition_seq, tgt_remi_seq, insts_to_pred, keep_melody=False
        )
        return condition_seq, tgt_remi_seq

    def aug_inst_pred_bass(self, condition_seq, tgt_remi_seq):
        '''
        Delete drum from condition seq, keep tgt as-is
        '''
        # print('pred_bass')
        insts_to_pred = ['i-32', 'i-33', 'i-43', 'i-70']
        condition_seq, tgt_remi_seq = self.aug_inst_pred_insts(
            condition_seq, tgt_remi_seq, insts_to_pred, keep_melody=True
        )
        return condition_seq, tgt_remi_seq

    def aug_inst_del_drum(self, condition_seq, tgt_remi_seq):
        '''
        Delete drum from 
        - inst spec in condition
        - target sequence

        NOTE: If other part of the condition sequence contains drum, keep it as-is
        '''
        # print('drum deletion aug')
        insts_to_del = ['i-128']
        condition_seq, tgt_remi_seq = self.aug_inst_del_specified_inst(condition_seq, tgt_remi_seq, insts_to_del)
        return condition_seq, tgt_remi_seq

    def aug_inst_tracks_infill(self, condition_seq, tgt_remi_seq):
        '''
        Reconstruct the content so that on average
        50% of the tracks are deleted from content, and the rest are kept
        '''
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq)
        if len(insts) <= 1:
            return condition_seq, tgt_remi_seq

        ''' Melody Keeping '''
        mel_inst = self.__get_melody_inst(tgt_remi_seq, insts)
        non_mel_insts = insts.copy()
        non_mel_insts.remove(mel_inst)

        # Determine the number of instruments to delete
        lamb = len(non_mel_insts) // 2
        num_inst_del = np.random.poisson(lamb)
        num_inst_del = min(num_inst_del, len(non_mel_insts))
        num_inst_del = max(num_inst_del, 1)

        # Randomly select instruments to delete
        insts_to_pred = random.sample(non_mel_insts, num_inst_del)

        condition_seq, tgt_remi_seq = self.aug_inst_pred_insts(
            condition_seq, 
            tgt_remi_seq, 
            insts_to_pred, 
        )

        return condition_seq, tgt_remi_seq

    def aug_inst_del_bass(self, condition_seq, tgt_remi_seq):
        # There are multiple instruments to be deleted. Delete them all.
        # Acoustic bass, electric bass, contrabass, bassoon.
        # ID: 32, 33, 43, 70
        # print('bass deletion aug')
        insts_to_del = ['i-32', 'i-33', 'i-43', 'i-70']
        condition_seq, tgt_remi_seq = self.aug_inst_del_specified_inst(condition_seq, tgt_remi_seq, insts_to_del)
        return condition_seq, tgt_remi_seq

    def aug_inst_del_specified_inst(self, condition_seq, tgt_remi_seq, insts_to_del):
        # Track deletion, in target sequence (changes will reflect in condition seq by construct it again)
        insts = remi_utils.from_remi_get_insts(tgt_remi_seq)
        num_insts = len(insts)
        if num_insts <= 1:
            return condition_seq, tgt_remi_seq

        new_insts = insts.copy()
        for inst_to_del in insts_to_del:
            if inst_to_del in insts:
                new_insts.remove(inst_to_del)

        # Modify the remi, delete notes do not needed.
        tgt_remi_seq = self.__retain_specified_insts_in_remi(tgt_remi_seq, new_insts) # NOTE: might be buggy for 2-bar segment
        new_insts = remi_utils.in_inst_list_sort_inst(list(new_insts))

        # Modify inst in conditions according to track deletion augmentation
        condition_seq = remi_utils.in_condition_keep_only_specified_insts(
            condition_seq, 
            new_insts,
            has_texture=self.texture_control
        )

        return condition_seq, tgt_remi_seq

    def reorder_tgt(self, remi_seq):
        '''
        Re-order the target sequence, so that it become track-by-track, instead of mixing together
        
        Notes in remi seq can be either
        - o i p d
        - i p d
        - p d

        In return:
        - i o p d o p d ...  i o p d p d o p d
        '''
        seq_of_inst = {}
        insts = remi_utils.from_remi_get_insts(remi_seq) # Get inst, sort by program id

        if len(remi_seq) > 1 and len(insts) == 0:
            insts = ['i-0']

        for inst in insts:
            seq_of_inst[inst] =  []

        pre_pos = None
        cur_pos = None
        pre_inst = None
        cur_inst = None
        cur_p = None
        cur_dur = None
        for tok in remi_seq:
            if tok.startswith('o-'):
                cur_pos = tok
            elif tok.startswith('i-'):
                cur_inst = tok
            elif tok.startswith('p-'):
                cur_p = tok
            elif tok.startswith('d-'):
                cur_dur = tok

                # If no instrument, set to the first instrument
                if cur_inst is None:
                    cur_inst = insts[0]

                # Add the note to its corresponding sequence
                if cur_inst != pre_inst and cur_inst is not None: # If for new inst
                    seq_of_inst[cur_inst].append(cur_pos)
                else: # If for a same instrument
                    if pre_pos is not None and cur_pos == pre_pos: # If for same position
                        pass # No need to add pos token
                    else:   # If for different position
                        seq_of_inst[cur_inst].append(cur_pos) # should add pos token
                seq_of_inst[cur_inst].append(cur_p)
                seq_of_inst[cur_inst].append(cur_dur)

                pre_pos = cur_pos
                pre_inst = cur_inst

        ret = []
        for inst in seq_of_inst:
            ret.append(inst)
            ret.extend(seq_of_inst[inst])

        return ret

    def __retain_specified_insts_in_remi(self, remi_seq, inst_to_preserve: list):
        '''
        Modify a remi_seq so that it only contains notes from specified instruments
        '''

        new_inst = set(inst_to_preserve)

        # Retain notes only for selected instruments
        new_remi = []
        pos_buffer = []
        retain = True
        for tok in remi_seq:
            if tok.startswith("o-"):
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                pos_buffer = [tok]
            elif tok.startswith("i-"):
                if tok in new_inst:  # continue until meet new instrument
                    retain = True
                    pos_buffer.append(tok)
                else:
                    retain = False
            elif tok.startswith("p-") or tok.startswith("d-"):
                if retain:
                    pos_buffer.append(tok)
            elif tok == "b-1":
                pos_buffer.append(tok)
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                    pos_buffer = [] # When multiple bars in segment, need to clear the pos_buffer at the end of each bar

        # Filter out empty position tokens
        ret = []
        for i, tok in enumerate(new_remi):
            if tok.startswith('o-') and new_remi[i+1].startswith('o-'): # If we see a position token
                # And if it's followed by another position token
                continue
            else:
                ret.append(tok) # append to result in other cases

        return ret
    
    def __remove_specified_insts_in_remi(self, remi_seq, inst_to_delete: list):
        '''
        Modify a remi_seq so that it only contains notes from specified instruments
        '''

        inst_to_delete = set(inst_to_delete)

        # Retain notes only for selected instruments
        new_remi = []
        pos_buffer = []
        retain = True
        for tok in remi_seq:
            if tok.startswith("o-"):
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                pos_buffer = [tok]
            elif tok.startswith("i-"):
                if tok not in inst_to_delete:  # continue until meet new instrument
                    retain = True
                    pos_buffer.append(tok)
                else:
                    retain = False
            elif tok.startswith("p-") or tok.startswith("d-"):
                if retain:
                    pos_buffer.append(tok)
            elif tok == "b-1":
                pos_buffer.append(tok)
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                    pos_buffer = [] # When multiple bars in segment, need to clear the pos_buffer at the end of each bar

        # Filter out empty position tokens
        ret = []
        for i, tok in enumerate(new_remi):
            if tok.startswith('o-') and new_remi[i+1].startswith('o-'): # If we see a position token
                # And if it's followed by another position token
                continue
            else:
                ret.append(tok) # append to result in other cases

        return ret

    def __get_random_pitch_token(self, upper=127) -> str:
        """Obtain a random pitch token

        Returns:
            str: A random pitch token in the supported vocab of MuseCoco (p-0 ~ p-255)
        """        
        p_value = random.randint(0, upper)
        ret = 'p-{}'.format(p_value)
        return ret
    
    def __get_random_position_token(self) -> str:
        """Obtain a random position token

        Returns:
            str: A random position token in the supported vocab of MuseCoco (o-0 ~ o-47) (majority)
        """     
        o_value = random.randint(0, 47)
        ret = 'o-{}'.format(o_value)
        return ret

    def __get_random_pitch_tokens(self, n_tok: int) -> List[str]:
        """Obtain a list of random pitch tokens

        Args:
            n_tok (int): the number of pitch tokens we want in the returned list.
        Returns:
            List[str]: a list of pitch tokens. len(return) == n_tok.
        """        
        ret = random.choices(range(256), k=n_tok)
        ret.sort(reverse=True)
        ret = ['p-{}'.format(i) for i in ret]
        return ret

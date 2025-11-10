'''
Do the generation with the test set
'''
import os
import sys
dirof = os.path.dirname
sys.path.append(dirof(dirof(os.path.abspath(__file__))))

if len(sys.argv) != 2: # For debug runs
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import re
import torch
from tqdm import tqdm
from torch import utils
from utils_common.utils import jpath, read_yaml, get_latest_checkpoint, ls
from utils_midi.utils_midi import RemiTokenizer, RemiUtil
from utils_instrument.inst_map import InstMapUtil
from lightning_dataset import *
from lightning_train import get_dataloader
from lightning.pytorch import seed_everything
from lightning_model import *
from transformers import AutoTokenizer
import mlconfig
from typing import List
from remi_z import MultiTrack, Bar

if __name__ == '__main__':
    seed_everything(42, workers=True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    infererence_test()
    # piano_reduc_infer_for_all()


def piano_reduc_infer_for_all():
    '''
    Do the inference on the entire test set using the piano reduction model
    '''
    # Specify paths
    config_fp = '/home/longshen/work/MuseCoco/musecoco/m2m/hparams/piano_reduction/reduction_dur_direct_range.yaml'
    pretrain_model_url = 'LongshenOu/m2m_pianist_dur'

    # Load the config, model, and tokenizer
    config = mlconfig.load(config_fp)  
    tk_fp = config['tokenizer_fp']
    tk = AutoTokenizer.from_pretrained(tk_fp, padding_side='left')
    lit_model = get_lit_model(pretrain_model_url, tk, config)
    model = lit_model.model
    model.cuda()

    test_set_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/original/test'
    out_dir = '/data2/longshen/musecoco_data/infer_out/piano_reduc/ours'
    song_names = ls(test_set_dir)
    for song_name in tqdm(song_names):
        song_dir = jpath(test_set_dir, song_name)
        midi_fp = jpath(song_dir, 'all_src.mid')
        out_fp = jpath(out_dir, f'{song_name}.mid')
        
        if os.path.exists(out_fp):
            continue

        # Run the inference
        arrange_for_midi(
            config, 
            model, 
            tk, 
            [], 
            midi_fp, 
            out_fp
        )


def infererence_test():
    '''
    Test the inference on a single midi
    '''
    # Specify paths
    config_fp = '/home/longshen/work/MuseCoco/musecoco/m2m/hparams/piano_reduction/piano_new.yaml'
    lit_ckpt = '/data2/longshen/musecoco_data/results_new/piano_arrange/bs16_lr0.0001_ep5_monitorf1/lightning_logs/version_0/checkpoints/epoch=03-valid_loss=0.39.ckpt'

    # Load the config, model, and tokenizer
    config = mlconfig.load(config_fp)  
    tk_fp = config['tokenizer_fp']
    tk = AutoTokenizer.from_pretrained(tk_fp, padding_side='left')
    lit_model = load_lit_model(tk, config, lit_ckpt)
    # get_lit_model(pretrain_model_url, tk, config)
    model = lit_model.model
    model.cuda()

    # Run the inference
    inp_fp = '/data2/longshen/musecoco_data/full_song/caihong/caihong.mid'
    # inp_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/original/test/Track01884/all_src.mid'
    out_fp = '/data2/longshen/musecoco_data/infer_out/piano_reduc/demos/caihong_new_q16.mid'
    # out_fp = '/data2/longshen/musecoco_data/infer_out/piano_reduc/demos/1884_new_k10_q16.mid'

    arrange_for_midi_new(
        config, 
        model, 
        tk, 
        [], 
        inp_fp, 
        out_fp,
        q16=True,
    )


def arrange_for_midi_new(config, model, tokenizer, new_insts:List[int], inp_midi_fp, out_midi_fp, q16=False):
    '''
    Arrange for a midi file
    Can be used for both band and piano arrangement model.
    Used the remi_z to handel MIDI files
    '''
    # Get model and tokenizer ready
    tk = tokenizer
    model.eval()

    # Config inference
    generate_kwargs = {
        # 'min_length': 500,
        'max_length': 800,
        'use_cache': True, 
        'bad_words_ids': [[tk.pad_token_id], [tk.convert_tokens_to_ids('[PAD]')], [tk.convert_tokens_to_ids('[INST]')], [tk.convert_tokens_to_ids('[SEP]')]],

        'no_repeat_ngram_size': config['no_repeat_token'] if 'no_repeat_token' in config else 6,
        
        # # Greedy decode
        # 'do_sample': False,

        # 'num_beams': 5,
        # 'do_sample': False,

        # Sampling
        'do_sample': True, # User searching method
        'top_k': config['top_k'],
        'top_p': config['top_p'],
        'temperature': config['temp'],

        # # Contrastive search
        # 'do_sample': False,
        # 'penalty_alpha': 0.6, #0.6
    }

    # Read the midi file
    mt = MultiTrack.from_midi(inp_midi_fp)
    mt.normalize_pitch()
    if q16:
        mt.quantize_to_16th()

    # Check time signature
    ts = mt.time_signatures
    if len(ts) > 1:
        print(f'Multiple time signatures in the midi file: {ts}')
        return
    if ts[0] != (4, 4):
        print(f'Non 4/4 time signature in the midi file: {ts}')
        return
    
    # Build single song dataset
    song_dataset = {}
    '''
    "Track00674-bar80": {
        "meta": {
            "time_signature": "(4, 4)",
            "tempo": 135.4,
            "insts": "4 26 35 48 52 69 100 126 128",
            "pitch_range": 109,
            "has_piano": true,
            "piano_pitch_range": 13,
            "has_drum": true
        },
        "content": "s-9 t-37 i-126 o-0 p-119 d-127 p-56 d-127 p-11 d-127 o-1 p-118 d-127 p-117 d-127 i-69 o-0 p-119 d-127 p-50 d-127 p-11 d-127 o-1 p-118 d-127 p-117 d-127 i-100 o-0 p-62 d-3 o-6 p-62 d-9 o-15 p-60 d-5 o-30 p-59 d-15 i-48 o-9 p-60 d-87 i-52 o-0 p-60 d-91 p-57 d-90 i-26 o-2 p-60 d-49 o-14 p-53 d-15 o-26 p-65 d-26 o-38 p-53 d-22 i-4 o-11 p-53 d-11 o-23 p-65 d-47 p-60 d-47 p-57 d-32 o-41 p-53 d-9 i-35 o-0 p-29 d-24 o-24 p-41 d-24 i-128 o-0 p-170 d-1 p-164 d-1 o-24 p-170 d-1 b-1",
        "hist": "Track00674-bar79"
    }
    '''
    piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])
    song_name = os.path.basename(inp_midi_fp).split('.')[0]
    for bar in mt.bars:
        bar_id = f'{song_name}-bar{bar.bar_id}'

        insts = list(bar.tracks.keys())
        insts.sort()
        insts = [str(i) for i in insts]
        insts_str = ' '.join(insts)

        song_dataset[bar_id] = {
            'meta': {
                'time_signature': ts[0],
                'tempo': bar.tempo,
                'insts': insts_str,
                'pitch_range': bar.get_pitch_range(),
                'has_piano': bar.has_piano(),
                'piano_pitch_range': bar.get_pitch_range(piano_ids),
                'has_drum': bar.has_drum(),
            },
            'content': ' '.join(bar.to_remiz_seq()),
            'hist': f'{song_name}-bar{bar.bar_id-1}' if bar.bar_id > 0 else None,
        }
    infer_dataset = {'infer': song_dataset}
    song_dataset_fp = '/data2/longshen/musecoco_data/infer_temp/song_dataset.json'
    save_json(infer_dataset, song_dataset_fp)

    dataset_class = config['dataset_class']
    dataset_class = eval(dataset_class)
    test_dataset = dataset_class(
        data_fp=song_dataset_fp, 
        split='infer', 
        config=config,
    )
    test_loader = utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=1,
    )

    # Config instruments
    if 'replace_inst' in config and config['replace_inst'] is not False:
        print('Inst setting: ', config['replace_inst'])

    # Iterate over dataset
    # NOTE: use the previous bar out as hist of next bar
    inputs = []
    test_out = []
    hist_remi = None
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for id, batch in enumerate(pbar):
            pbar.set_description(str(id))

            # Get the total seq (input and target)
            tot_seq = batch[0].strip().split(' ')
            sep_idx = tot_seq.index('[SEP]')
            inp_seq = tot_seq[:sep_idx+1]

            # Get the batch, replace the hist
            if config['hist_autoregressive']:
                if hist_remi != None:

                    # Replace the hist subseq
                    hist_start_idx = inp_seq.index('[HIST]') + 1
                    hist_end_idx = inp_seq.index('[SEP]')
                    inp_seq = inp_seq[:hist_start_idx] + hist_remi + inp_seq[hist_end_idx:]

            # Replace the instrument
            if 'replace_inst' in config and config['replace_inst'] is not False:
                inst_spec = config['replace_inst']

                inst_start_idx = inp_seq.index('[INST]') + 1
                if '[MELODY]' not in inp_seq:
                    inst_end_idx = inp_seq.index('[PITCH]')
                else:
                    inst_end_idx = inp_seq.index('[MELODY]')
                inp_seq = inp_seq[:inst_start_idx] + inst_spec + inp_seq[inst_end_idx:]

            # Prepare input string
            inp_str = ' '.join(inp_seq)
            inputs.append(inp_str)

            # Tokenize the batch
            batch_tokenized = tk(
                [inp_str], 
                return_tensors="pt",
                padding=False,
                add_special_tokens=False, # Don't add eos token
            )['input_ids'].cuda()

            # Feed to the model
            out = model.generate(
                batch_tokenized,
                pad_token_id=tk.pad_token_id,
                **generate_kwargs
            )
            out_str = tk.batch_decode(out)[0] # because we do bs=1 here

            # Select substr between [SEP] and [EOS] as output
            out_str = out_str.split('[SEP]')[1].split('[EOS]')[0].strip()

            # Truncate only useful part, i.e., between <sep> and </s>
            out_seq = out_str.split(' ')

            # Ensure the output part is a bar
            if out_seq[-1] != 'b-1' and out_seq.count('b-1') == 0:
                out_seq.append('b-1')
            
            # Truncate content more than 1 bar
            bar_idx = out_seq.index('b-1')
            out_seq = out_seq[:bar_idx+1]


            hist_remi = out_seq
            test_out.extend(out_seq)

    # Clean the output
    new_out = []
    for tok in test_out:
        t = tok.strip()
        if len(t) > 0:
            new_out.append(t)
    test_out = new_out

    # Save output
    out_mt = MultiTrack.from_remiz_seq(test_out)
    out_mt.to_midi(out_midi_fp)


def arrange_for_midi(config, model, tokenizer, new_insts:List[int], inp_midi_fp, out_midi_fp):
    '''
    Arrange for a midi file
    Can be used for both band and piano arrangement model.
    '''
    # Get model and tokenizer ready
    tk = tokenizer
    model.eval()

    # Config inference
    generate_kwargs = {
        # 'min_length': 500,
        'max_length': 800,
        'use_cache': True, 
        'bad_words_ids': [[tk.pad_token_id], [tk.convert_tokens_to_ids('[PAD]')], [tk.convert_tokens_to_ids('[INST]')], [tk.convert_tokens_to_ids('[SEP]')]],

        'no_repeat_ngram_size': config['no_repeat_token'] if 'no_repeat_token' in config else 6,
        
        # # Greedy decode
        # 'do_sample': False,

        # 'num_beams': 5,
        # 'do_sample': False,

        # Sampling
        'do_sample': True, # User searching method
        'top_k': config['top_k'],
        'top_p': config['top_p'],
        'temperature': config['temp'],

        # # Contrastive search
        # 'do_sample': False,
        # 'penalty_alpha': 0.6, #0.6
    }

    # Generate remi seqs, save to song_seq_remi.txt
    remi_tk = RemiTokenizer()
    song_remi_seq = remi_tk.midi_to_remi(
            inp_midi_fp, 
            return_pitch_shift=False, 
            return_key=False, 
            normalize_pitch=True, 
            reorder_by_inst=True,
            include_ts=False,
            include_tempo=False,
            include_velocity=False,
    )

    # Remove drum from the remi seq
    inst_util = InstMapUtil()
    remi_seq_new = []
    bar_indices = remi_utils.from_remi_get_bar_idx(song_remi_seq)
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = song_remi_seq[bar_start_idx:bar_end_idx]
        opd_seqs = remi_utils.from_remi_get_opd_seq_per_track(bar_seq)
        
        # Reconstruct the bar_seq
        bar_seq_new = []
        insts_with_voice = remi_utils.from_remi_get_inst_and_voice(bar_seq)
        if 'i-128' in insts_with_voice:
            insts_with_voice.remove('i-128')
        for inst in insts_with_voice:

            # Quantize instrument
            inst_id = int(inst.split('-')[1])
            inst_id_quant = inst_util.slakh_quantize_inst_prog(inst_id)
            if inst_id_quant is None:
                inst_id_quant = inst_id
                # continue
            inst_tok = 'i-{}'.format(inst_id_quant)

            bar_seq_new.append(inst_tok)
            bar_seq_new.extend(opd_seqs[inst])
        bar_seq_new.append('b-1')

        remi_seq_new.extend(bar_seq_new)
    song_remi_seq = remi_seq_new

    # Split to bars
    infer_temp_dir = '/data2/longshen/musecoco_data/infer_temp'
    create_dir_if_not_exist(infer_temp_dir)
    seg_remi_seqs = remi_utils.song_remi_split_to_segments_2bar_hop1(song_remi_seq, ts_and_tempo=False)
    seg_remi_seqs_fp = jpath(infer_temp_dir, 'seg_remi.txt')
    t = [' '.join(i) + '\n' for i in seg_remi_seqs]
    with open(seg_remi_seqs_fp, 'w') as f:
        f.writelines(t)

    # Construct dataset
    '''
    "Track00674-bar80": {
        "meta": {
            "time_signature": "(4, 4)",
            "tempo": 135.4,
            "insts": "4 26 35 48 52 69 100 126 128",
            "pitch_range": 109,
            "has_piano": true,
            "piano_pitch_range": 13,
            "has_drum": true
        },
        "content": "s-9 t-37 i-126 o-0 p-119 d-127 p-56 d-127 p-11 d-127 o-1 p-118 d-127 p-117 d-127 i-69 o-0 p-119 d-127 p-50 d-127 p-11 d-127 o-1 p-118 d-127 p-117 d-127 i-100 o-0 p-62 d-3 o-6 p-62 d-9 o-15 p-60 d-5 o-30 p-59 d-15 i-48 o-9 p-60 d-87 i-52 o-0 p-60 d-91 p-57 d-90 i-26 o-2 p-60 d-49 o-14 p-53 d-15 o-26 p-65 d-26 o-38 p-53 d-22 i-4 o-11 p-53 d-11 o-23 p-65 d-47 p-60 d-47 p-57 d-32 o-41 p-53 d-9 i-35 o-0 p-29 d-24 o-24 p-41 d-24 i-128 o-0 p-170 d-1 p-164 d-1 o-24 p-170 d-1 b-1",
        "hist": "Track00674-bar79"
    }
    '''
    bs = 1
    split = 'infer'
    dataset_class = config['dataset_class'] if 'dataset_class' in config else 'ArrangerDataset'
    dataset_class = eval(dataset_class)
    test_dataset = dataset_class(
        data_fp=seg_remi_seqs_fp, 
        split=split, 
        config=config,
    )
    test_loader = utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=bs,
    )

    # Config instruments
    if 'replace_inst' in config and config['replace_inst'] is not False:
        print('Inst setting: ', config['replace_inst'])

    # Iterate over dataset
    # NOTE: use the previous bar out as hist of next bar
    inputs = []
    test_out = []
    hist_remi = None
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for id, batch in enumerate(pbar):
            pbar.set_description(str(id))

            # Get the total seq (input and target)
            tot_seq = batch[0].strip().split(' ')
            sep_idx = tot_seq.index('[SEP]')
            inp_seq = tot_seq[:sep_idx+1]

            # Get the batch, replace the hist
            if config['hist_autoregressive']:
                if hist_remi != None:

                    # Replace the hist subseq
                    hist_start_idx = inp_seq.index('[HIST]') + 1
                    hist_end_idx = inp_seq.index('[SEP]')
                    inp_seq = inp_seq[:hist_start_idx] + hist_remi + inp_seq[hist_end_idx:]

            # Replace the instrument
            if 'replace_inst' in config and config['replace_inst'] is not False:
                inst_spec = config['replace_inst']

                inst_start_idx = inp_seq.index('[INST]') + 1
                if '[MELODY]' not in inp_seq:
                    inst_end_idx = inp_seq.index('[PITCH]')
                else:
                    inst_end_idx = inp_seq.index('[MELODY]')
                inp_seq = inp_seq[:inst_start_idx] + inst_spec + inp_seq[inst_end_idx:]

            # Prepare input string
            inp_str = ' '.join(inp_seq)
            inputs.append(inp_str)

            # Tokenize the batch
            batch_tokenized = tk(
                [inp_str], 
                return_tensors="pt",
                padding=False,
                add_special_tokens=False, # Don't add eos token
            )['input_ids'].cuda()

            # Feed to the model
            out = model.generate(
                batch_tokenized,
                pad_token_id=tk.pad_token_id,
                **generate_kwargs
            )
            out_str = tk.batch_decode(out)[0] # because we do bs=1 here

            # Select substr between [SEP] and [EOS] as output
            out_str = out_str.split('[SEP]')[1].split('[EOS]')[0].strip()

            # Truncate only useful part, i.e., between <sep> and </s>
            out_seq = out_str.split(' ')

            # Ensure the output part is a bar
            if out_seq[-1] != 'b-1' and out_seq.count('b-1') == 0:
                out_seq.append('b-1')
            
            # Truncate content more than 1 bar
            bar_idx = out_seq.index('b-1')
            out_seq = out_seq[:bar_idx+1]


            hist_remi = out_seq
            test_out.extend(out_seq)

    # Clean the output
    new_out = []
    for tok in test_out:
        t = tok.strip()
        if len(t) > 0:
            new_out.append(t)
    test_out = new_out

    # Save output
    remi_tk.remi_to_midi(test_out, out_midi_fp)



if __name__ == '__main__':
    main()
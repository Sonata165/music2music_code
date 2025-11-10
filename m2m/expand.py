'''
Do full-song arrangement from lead sheet
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')
if __name__ == '__main__': # Debug
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from lightning.pytorch import seed_everything
from m2m.infer import get_latest_checkpoint
from torch import utils
from utils_midi.utils_midi import RemiTokenizer, RemiUtil
from utils_midi import remi_utils
from utils_common.utils import jpath, read_yaml, create_dir_if_not_exist
from m2m.lightning_dataset import *
from m2m.lightning_model import *
from utils_instrument.inst_map import InstMapUtil
import mlconfig
from utils_chord import chord_detect_from_remi as chord_utils

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    # prepare_chord_ref()
    generate_arrangement()


def prepare_chord_ref():
    '''
    Generate a chord reference file for the song
    Using the ExpansionDataset
    '''
    seed_everything(42, workers=True)

    if len(sys.argv) != 2:
        config_fp = '/home/longshen/work/musecoco/m2m/hparams/expand/chord_note.yaml'
        config = mlconfig.load(config_fp)
    else:
        config_fp = sys.argv[1]
        assert os.path.exists(config_fp), 'Config file not found'
        config = mlconfig.load(config_fp)
        
    # Prepare MIDI file
    midi_fp_dict = read_yaml('../utils_arrange/song_path.yaml')
    # midi_fp_dict = read_yaml('utils_arrange/song_path.yaml')
    assert config['song_name'] in midi_fp_dict, 'Song not found in the song_path.yaml'
    midi_fp = midi_fp_dict[config['song_name']]

    # Prepare paths
    exp_name = config['reinst_exp_name']
    save_dir = os.path.dirname(midi_fp)
    save_fp = jpath(save_dir, 'chord_ref.txt')

    # Generate remi seqs, save to song_seq_remi.txt
    remi_tk = RemiTokenizer()
    midi_dir_name = os.path.dirname(midi_fp)
    song_remi_seq = remi_tk.midi_to_remi(
            midi_fp, 
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
                continue
            inst_tok = 'i-{}'.format(inst_id_quant)

            bar_seq_new.append(inst_tok)
            bar_seq_new.extend(opd_seqs[inst])
        bar_seq_new.append('b-1')

        remi_seq_new.extend(bar_seq_new)
    song_remi_seq = remi_seq_new
    # RemiUtil.save_remi_seq(song_remi_seq, remi_fp)
    # detokenized_fp = jpath(midi_dir_name, "detokenized.mid")
    # remi_tk.remi_to_midi(song_remi_seq, detokenized_fp)
    seg_remi_seqs = remi_utils.song_remi_split_to_segments_2bar_hop1(song_remi_seq, ts_and_tempo=False)
    seg_remi_seqs_fp = jpath(midi_dir_name, 'song_seg.txt')
    t = [' '.join(i) + '\n' for i in seg_remi_seqs]
    with open(seg_remi_seqs_fp, 'w') as f:
        f.writelines(t)

    chords = []
    for seg_remi_seq in seg_remi_seqs:
        print(seg_remi_seq)
        _, tgt_seq = remi_utils.from_remi_two_bar_split_hist_tgt_seq(seg_remi_seq)
        chord_seq = remi_utils.from_remi_get_chord_seq(tgt_seq)
        
        chord_of_bar = []
        for chord in chord_seq:
            if chord is None:
                chord_of_bar.append('N')
            else:
                root, type = chord
                chord_of_bar.append('{}:{}'.format(root, type))
        chords.append(chord_of_bar)

    # Write to file
    with open(save_fp, 'w') as f:
        for chord_of_bar in chords:
            f.write(' '.join(chord_of_bar) + '\n')

    # Save the melody
    melody_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/infer_input/full_song/caihong/caihong_melody.mid'
    melody_remi_seq = remi_tk.midi_to_remi(
            melody_fp, 
            return_pitch_shift=False, 
            return_key=False, 
            normalize_pitch=True, 
            reorder_by_inst=True,
            include_ts=False,
            include_tempo=False,
            include_velocity=False,
    )
    seg_remi_seqs = remi_utils.song_remi_split_to_segments_2bar_hop1(melody_remi_seq, ts_and_tempo=False)
    seg_remi_seqs_fp = jpath(midi_dir_name, 'caihong_melody.txt')
    t = [' '.join(i) + '\n' for i in seg_remi_seqs]
    with open(seg_remi_seqs_fp, 'w') as f:
        f.writelines(t)



def generate_arrangement():
    seed_everything(42, workers=True)

    if len(sys.argv) != 2:
        config_fp = '/home/longshen/work/musecoco/m2m/hparams/expand/chord_note.yaml'
        config = mlconfig.load(config_fp)
        
    else:
        config_fp = sys.argv[1]
        assert os.path.exists(config_fp), 'Config file not found'
        config = mlconfig.load(config_fp)
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fp_dict = read_yaml(jpath(cur_dir, '../utils_arrange/lead_sheet_path.yaml'))
    # fp_dict = read_yaml('utils_arrange/lead_sheet_path.yaml')
    
    # Preprae melody (in txt) and chord (in txt)
    assert config['song_name'] in fp_dict, 'Song not found in the lead_sheet_path.yaml'
    song_name = config['song_name']
    melody_fp = fp_dict[song_name]['melody']
    chord_fp = fp_dict[song_name]['chord']
    assert os.path.exists(melody_fp), 'Melody file not found'
    assert os.path.exists(chord_fp), 'Chord file not found'
    
    # Check chord legality
    with open(chord_fp, 'r') as f:
        lines = f.readlines()
    for line in lines:
        assert len(line.strip().split(' ')) == 4, 'Chord format not correct'
        chords = line.strip().split(' ')
        for chord in chords:
            if chord != 'N':
                root, type = chord.split(':')
                assert root in chord_utils.note_name_to_note_id.keys(), 'Root not found in the root map'
                assert type in chord_utils.chords.keys(), 'Type not found in the type map'

    # Prepare paths
    exp_name = config['reinst_exp_name']

    # Load the model and tokenizer
    tk_fp = config['tokenizer_fp']
    tk = AutoTokenizer.from_pretrained(tk_fp)
    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    pt_ckpt = config['pt_ckpt']
    model_cls = eval(config['lit_model_class'])
    l_model = model_cls.load_from_checkpoint(ckpt_fp, pt_ckpt=pt_ckpt, tokenizer=tk, infer=True) # TODO: change to model_fp
    model = l_model.model
    model.eval()

    # Construct dataset, 
    bs = 1
    split = 'infer'
    dataset_class = 'ExpanderInferDataset'
    dataset_class = eval(dataset_class)
    test_dataset = dataset_class(
        melody_fp=melody_fp, 
        chord_fp=chord_fp,
        split=split, 
        config=config,
    )
    test_loader = utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=bs,
    )

    t = tk.pad_token

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

            hist_remi = out_seq
            test_out.extend(out_seq)

    # Clean the output
    new_out = []
    for tok in test_out:
        t = tok.strip()
        if len(t) > 0:
            new_out.append(t)
    test_out = new_out

    # Prepare saving folders
    save_dir = jpath(out_dir, 'lightning_logs', latest_version_dir, 'infer')
    midi_out_dir = jpath(save_dir, 'out_midi', config['reinst_group'])
    token_inp_dir = jpath(save_dir, 'inp_token', config['reinst_group'])
    token_out_dir = jpath(save_dir, 'out_token', config['reinst_group'])
    create_dir_if_not_exist(midi_out_dir)
    create_dir_if_not_exist(token_inp_dir)
    create_dir_if_not_exist(token_out_dir)

    # Save input tokens
    save_fn = '{}.txt'.format(exp_name)
    input_save_fp = jpath(token_inp_dir, save_fn)
    with open(input_save_fp, 'w') as f:
        f.writelines([i+'\n' for i in inputs])

    # Save output tokens
    save_fn = '{}.txt'.format(exp_name)
    out_remi_fp = jpath(token_out_dir, save_fn)
    RemiUtil.save_remi_seq(test_out, out_remi_fp)

    # Save midi
    save_fn = '{}.mid'.format(exp_name)
    out_midi_fp = jpath(midi_out_dir, save_fn)
    remi_tk = RemiTokenizer()
    remi_tk.remi_to_midi(test_out, out_midi_fp)


if __name__ == '__main__':
    main()
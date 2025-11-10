
import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(__file__)))

from torch.utils.data import Dataset, DataLoader
from m2m.datasets_new.PianoReducDataset import PianoReducDataset
# from src_hf.lightning_dataset import *
# from src_hf.utils import jpath, read_yaml
from tqdm import tqdm
from transformers import MuseCocoTokenizer
from utils_midi import remi_utils
import mlconfig

# config_fn = 'arrangement/reorder_vc_rawhist.yaml'
# config_fn = 'chord_pred/chord_pred.yaml'
# split = 'valid'

# config_fp = jpath('../src_hf/hparams', config_fn)
# config = read_yaml(config_fp)
# data_root = config['data_root']
# data_fn = '{}.txt'.format(split)
# data_fp = jpath(data_root, data_fn)

def main():
    # test_dataset_general()
    # test_inst_pred_dataset()
    test_piano_new_dataset()


def test_piano_new_dataset():
    data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json'
    split = 'test'
    config_fp = '/home/longshen/work/MuseCoco/musecoco/m2m/hparams/piano_reduction/reduction_dur_direct_range.yaml'
    config = mlconfig.load(config_fp)
    dataset = PianoReducDataset(
        data_fp=data_fp, 
        split=split, 
        config=config,
    )

    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=32,
        num_workers=16,
        collate_fn=lambda x: x,
    )

    # tk_fp = config['tokenizer_fp']
    # tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    for id, batch in enumerate(tqdm(test_loader)):
        # Get the batch
        t = batch

        # # Get the note sequence and label
        # note_seqs = [' '.join(i[0]) for i in batch]
        # labels = [i[1] for i in batch]

        # # Tokenize the batch
        # batch_tokenized = tk(
        #     note_seqs, 
        #     return_tensors="pt",
        #     padding=True,
        # )['input_ids']

        # # Add BOS token
        # bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=tk.bos_token_id, dtype=torch.long)
        # batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        a = 1


def test_chord_pred_dataset():
    dataset = ChordPredDataset(data_fp=data_fp, split=split, config=config)

    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=5,
        collate_fn=lambda x: x,
    )

    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    for id, batch in enumerate(tqdm(test_loader)):
        # Get the batch
        t = batch

        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = [i[1] for i in batch]

        # Tokenize the batch
        batch_tokenized = tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids']

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=tk.bos_token_id, dtype=torch.long)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        a = 1


def test_inst_pred_dataset():
    dataset = InstPredDataset(data_fp=data_fp, split=split, config=config)

    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=5,
        collate_fn=lambda x: x,
    )

    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    for id, batch in enumerate(tqdm(test_loader)):
        # Get the batch
        t = batch

        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = [i[1] for i in batch]

        # Tokenize the batch
        batch_tokenized = tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids']

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=tk.bos_token_id, dtype=torch.long)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        a = 1


def test_dataset_general():
    # config['do_augment'] = False

    # dataset = ArrangerDataset(data_fp=data_fp, split=split, config=config)
    # dataset = ExpansionDataset(data_fp=data_fp, split=split, config=config)
    dataset = InstPredDataset(data_fp=data_fp, split=split, config=config)

    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=5,
    )

    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    for id, batch in enumerate(tqdm(test_loader)):
        # Get the batch
        t = batch

        # Tokenize the batch
        batch_tokenized = tk(
            batch, 
            return_tensors="pt",
            padding=True,
        )['input_ids']

        # Feed to the model
        a = 1

    
def get_pos_seq_from_condition_and_tgt_seq(tot_seq):
    condition, tgt = tot_seq.split(' <sep> ')
    condition = condition.strip().split(' ')
    tgt = tgt.strip().split(' ')

    t1 = [t for t in condition if t.startswith('o')]
    t2 = [t for t in tgt if t.startswith('o')]
    
    return t1, t2

def get_inst_seq_from_condition_and_tgt_seq(tot_seq):
    condition, tgt = tot_seq.split(' <sep> ')
    condition = condition.strip().split(' ')
    tgt = tgt.strip().split(' ')

    t1 = set([t for t in condition if t.startswith('i')])
    t2 = set([t for t in tgt if t.startswith('i')])
    
    return t1, t2

if __name__ == '__main__':
    main()
    
import os
import sys

sys.path.append('.')
sys.path.append(os.path.abspath('..'))
if not len(sys.argv) == 2 and __name__ == '__main__': # For debug runs
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch

from lightning.pytorch import seed_everything
from utils_common.utils import jpath, read_yaml
from m2m.lightning_dataset import *
from m2m.lightning_model import get_lit_model
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    seed_everything(42, workers=True)

    if not len(sys.argv) == 2: # Debug
        config_fp = '/home/longshen/work/MuseCoco/musecoco/m2m/hparams/piano_reduction/piano_new.yaml'
        config = read_yaml(config_fp)
        config['num_workers'] = 0
        config['fast_dev_run'] = 10
        # config['train_with'] = 'valid'
    else:
        config_fp = sys.argv[1]
        config = read_yaml(config_fp)

    # Init the model
    model_fp = config['pt_ckpt']
    tk_fp = config['tokenizer_fp']
    tk = AutoTokenizer.from_pretrained(tk_fp)

    lit_model = get_lit_model(model_fp, tk, config)

    # Setup data
    train_loader = get_dataloader(config, config['train_with'] if 'train_with' in config else 'train')
    valid_loader = get_dataloader(config, 'valid')

    # Train the model
    out_dir = jpath(config['result_root'], config['out_dir'])
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        mode="min",
        filename='{epoch:02d}-{valid_loss:.2f}',
        save_top_k=1,
    )
    earlystop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=config['early_stop_patience'],
        mode='min',
    )
    trainer = L.Trainer(
        max_epochs=config['n_epoch'],
        default_root_dir=out_dir, # output and log dir
        callbacks=[checkpoint_callback, earlystop_callback],
        fast_dev_run=config['fast_dev_run'] if 'fast_dev_run' in config else False,
        val_check_interval=config['val_check_interval'],
        precision='bf16',
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader,
    )

def get_dataloader(config, split):
    if split != 'test':
        bs = config['bs']
    else:
        bs = config['bs_test']
    data_root = config['data_root']
    data_fn = '{}.txt'.format(split)
    data_fp = jpath(data_root, data_fn)

    dataset_class_name = config['dataset_class']
    dataset_class = eval(dataset_class_name)

    dataset = dataset_class(data_fp=data_fp, split=split, config=config)
    dataloader = utils.data.DataLoader(
        dataset=dataset, 
        batch_size=bs,
        num_workers=config['num_workers'] if 'num_workers' in config else 4,
        collate_fn=lambda x: x,
    )
    return dataloader



if __name__ == '__main__':
    main()
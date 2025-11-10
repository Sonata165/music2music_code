'''
See if the model is still working after move model definition to the project folder
'''

import os
import sys

sys.path.append('..')

from src_hf.utils import read_yaml
from src_hf.lightning_model import LitMuseCoco
from hf_musecoco.modeling_musecoco import MuseCocoBarClsModelDoubleHead, MuseCocoPhraseClsModelSingleHead
from hf_musecoco.tokenization_musecoco import MuseCocoTokenizer


def main():
    test_cls_model()


def test_cls_model():
    config_fp = '/home/longshen/work/MuseCoco/musecoco/src_hf/hparams/inst_pred/inst_pred.yaml'
    config = read_yaml(config_fp)
    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)
    # model = MuseCocoBarClsModelDoubleHead(config['pt_ckpt'])

    n_labels = 35 # 35 types of instruments
    model = MuseCocoPhraseClsModelSingleHead(config['pt_ckpt'], n_labels=n_labels)

    sample_text = [
        's-7 o-0 i-0 p-4 d-5 o-8 i-2 p-7 d-20 b-1',
        's-7 o-0 i-0 p-4 d-5 b-1'
    ]
    input_ids = tk(sample_text, return_tensors='pt', padding=True)['input_ids']
    out = model(input_ids) # [bs, num_labels]
    print(out)


def test_local_script():
    
    config_fp = '/home/longshen/work/MuseCoco/musecoco/src_hf/hparams/expanding/reorder_vc_rawhist.yaml'
    config = read_yaml(config_fp)
    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    model_fp = config['pt_ckpt']
    lit_model = LitMuseCoco(
        model_fp,
        # model_config=model_config,
        tokenizer=tk, 
        hparams=config
    )




if __name__ == "__main__":
    main()
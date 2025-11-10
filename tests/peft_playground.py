import os
import sys

sys.path.append('..')

import torch
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from src_hf.utils import jpath, read_json
from transformers import MuseCocoLMHeadModel, MuseCocoConfig

def main():
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=4, 
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=['v_proj'], # try 3 settings: v, qv, qkv
    )

    # Initialize musecoco
    pt_ckpt = '/data1/longshen/musecoco_data/pretrained_models/1b/model'
    config_fp = jpath(pt_ckpt, 'config.json')
    config = read_json(config_fp)
    config = MuseCocoConfig.from_pretrained(config_fp)
    model = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

if __name__ == '__main__':
    main()
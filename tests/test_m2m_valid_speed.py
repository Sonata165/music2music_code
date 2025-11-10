'''
Check the model's speed on validation set
'''

import os
import sys

sys.path.append('..')

import torch
from src_hf.utils import jpath
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("LongshenOu/m2m_pt", torch_dtype=torch.bfloat16).cuda()
tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")

data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/slakh_8bar/valid.txt'
with open(data_fp, 'r') as f:
    data = f.readlines()
data = [l.strip() for l in data]

bs = 3
for i in tqdm(range(0, len(data), bs)):
    batch = data[i:i+bs]
    inp = tk(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.cuda()
    out = model(inp)



import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
sys.path.append('.')
sys.path.append('..')

from src_hf.utils import jpath
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from m2m.lightning_model import *
from check_output import convert_remi_to_midi

def main():
    test_conditional()

def test_conditional():
    # # Full seq loss
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/8bar_gen/lr1e-5_full_seq_loss/lightning_logs/version_1/checkpoints/epoch=01-valid_loss=0.63.ckpt'
    
    # # # tgt seq loss
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/8bar_gen/lr1e-5_tgt_seq_loss/lightning_logs/version_3/checkpoints/epoch=01-valid_loss=0.63.ckpt'

    # # inst aug
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/8bar_gen/lr1e-5_tgt_loss_auginst_fixed/lightning_logs/version_1/checkpoints/epoch=00-valid_loss=0.57.ckpt'

    # # Numbered bar
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/8bar_gen/lr1e-5_tgt_loss_auginst_numbered_bar/lightning_logs/version_0/checkpoints/epoch=01-valid_loss=0.60.ckpt'

    # Inst conditioned 4-bar gen
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/4bar_gen_inst_control/baseline/fullseqloss_bs6_lr1e-4/lightning_logs/version_0/checkpoints/epoch=01-valid_loss=0.65.ckpt'

    # # Inst conditioned 4-bar gen (inst aug)
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/4bar_gen_inst_control/auginst/lightning_logs/version_2/checkpoints/epoch=00-valid_loss=0.71.ckpt'

    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")
    # model = AutoModelForCausalLM.from_pretrained(ckpt_fp)
    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model
    model.eval()
    a = 1

    inp = tk.encode("[BOS] s-9 t-30 [INST] i-128 [SEP]", return_tensors="pt", add_special_tokens=False).cuda()

    print(inp)

    text = model.generate(
        inputs=inp,
        max_length=2048, 
        do_sample=True, 
        # num_beams=5,
        # top_k=50,
        # top_p=0.95,
        # temperature=0.6,
        # no_repeat_ngram_size=8,
    )
    # print(text)

    out = tk.decode(text[0], skip_special_tokens=False)

    # Remove condition and [EOS] from out
    out = out.split('[SEP]')[1].split('[EOS]')[0].strip()

    out_dir = '/home/longshen/work/musecoco/_misc'
    out_fp = jpath(out_dir, 'out.txt')
    with open(out_fp, 'w') as f:
        f.write(out)

    print('Converting to MIDI...')
    convert_remi_to_midi()
    print('Finish!')


def test_unconditional():

    model = AutoModelForCausalLM.from_pretrained("m2m_pt")
    tk = AutoTokenizer.from_pretrained("m2m_pt")
    # print(model)

    model.eval()
    model.cuda()

    inp = tk.encode("[BOS] s-9 t-36 i-0", return_tensors="pt", add_special_tokens=False).cuda()

    text = model.generate(
        inputs=inp,
        max_length=2048, 
        do_sample=True, 
        # top_k=50,
        # top_p=0.95,
    )
    # print(text)

    out = tk.decode(text[0], skip_special_tokens=True)
    # print(out)
    # print(tk.convert_ids_to_tokens(141))

    out_dir = '/home/longshen/work/MuseCoco/musecoco/temp'
    out_fp = jpath(out_dir, 'out.txt')
    with open(out_fp, 'w') as f:
        f.write(out)

    print('Finish!')

if __name__ == '__main__':
    main()
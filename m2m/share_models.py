import os
import sys
sys.path.append('../')
from transformers import AutoModelForCausalLM, AutoTokenizer
from m2m.lightning_model import *


def main():
    share_elab_model_no_pt()


def procedures():
    share_4_bar_inst_control()
    share_4_bar_inst_control_slakh()
    share_reinst_model()
    share_elab_model()
    share_reduction_model()
    share_drum_model()
    share_expand_model()
    share_reduction_dur_model()

    share_elab_model_no_pt()


def share_reduction_dur_model():
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/reduction_ret_aug_dur/ep3_lr1e-4_linear/lightning_logs/version_0/checkpoints/epoch=02-valid_loss=0.27.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    repo_name = "LongshenOu/m2m_pianist_dur"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_expand_model():
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/expansion/ep3_lr1e-4_linear/lightning_logs/version_4/checkpoints/epoch=02-valid_loss=0.77.ckpt'
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/expansion/four_chord/ep5_lr1e-4_linear/lightning_logs/version_0/checkpoints/epoch=03-valid_loss=0.71.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    repo_name = "LongshenOu/m2m_expander"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_drum_model():
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_8bar_quant_44/drum_arrange_4bar/ep3_lr1e-4_linear/lightning_logs/version_1/checkpoints/epoch=02-valid_loss=0.18.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    repo_name = "LongshenOu/m2m_drummer"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_reduction_model():
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/reduction_ret_aug/ep3_lr1e-4_linear/lightning_logs/version_4/checkpoints/epoch=02-valid_loss=0.69.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    repo_name = "LongshenOu/m2m_pianist"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_elab_model_no_pt():
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/elaboration/no_pt/ep10_lr1e-4_linear/lightning_logs/version_0/checkpoints/epoch=09-valid_loss=0.96.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    # repo_name = "LongshenOu/m2m_elaborator"
    repo_name = "LongshenOu/m2m_arranger_nopt"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_elab_model():
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/elaboration/ep3_lr1e-4_linear/lightning_logs/version_4/checkpoints/epoch=02-valid_loss=0.57.ckpt'
    # The new elab model with duration input
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/elaboration_dur/ep3_lr1e-4_linear/lightning_logs/version_0/checkpoints/epoch=02-valid_loss=0.19.ckpt'
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/elaboration/direct/ep3_lr1e-4_linear/lightning_logs/version_0/checkpoints/epoch=02-valid_loss=0.54.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    # repo_name = "LongshenOu/m2m_elaborator"
    repo_name = "LongshenOu/m2m_arranger"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_reinst_model():
    # Inst conditioned 4-bar gen
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/slakh_2bar_quant_44/arrange_1bar/reinst/ep3_lr1e-4_linear/lightning_logs/version_1/checkpoints/epoch=02-valid_loss=0.55.ckpt'
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")
    repo_name = "LongshenOu/m2m_reinst"

    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_4_bar_inst_control_slakh():
    # Inst conditioned 4-bar gen
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/4bar_gen_inst_control/baseline/bs6_fullseqloss_lr1e-4_bs12_ep50/lightning_logs/version_3/checkpoints/epoch=04-valid_loss=0.61.ckpt'

    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")
    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    repo_name = "LongshenOu/4-bar_inst-voice-control_slakh"

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)


def share_4_bar_inst_control():
    # Inst conditioned 4-bar gen
    ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/4bar_gen_inst_control/baseline/bs6_fullseqloss_lr1e-4/lightning_logs/version_0/checkpoints/epoch=01-valid_loss=0.65.ckpt'

    # # Inst conditioned 4-bar gen (inst aug)
    # ckpt_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/m2m_results/4bar_gen_inst_control/auginst/lightning_logs/version_2/checkpoints/epoch=00-valid_loss=0.71.ckpt'

    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")
    # model = AutoModelForCausalLM.from_pretrained(ckpt_fp)
    lit_model = LitM2mLM.load_from_checkpoint(ckpt_fp, 
                                              pt_ckpt='LongshenOu/m2m_pt',
                                              tokenizer=tk, 
                                              infer=True)
    model = lit_model.model

    repo_name = "LongshenOu/4-bar_inst-voice-control"

    model.push_to_hub(repo_name)
    tk.push_to_hub(repo_name)
    


if __name__ == '__main__':
    main()

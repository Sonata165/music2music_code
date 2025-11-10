"""
Do full-song reinstrumentation with HF model
Old version
Not very readable
"""

import os
import sys

sys.path.append(".")
sys.path.append("..")
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    seed_everything(42, workers=True)

    if len(sys.argv) != 2:
        print('Debug mode')
        default_hparam_fp = "/home/longshen/work/musecoco/m2m/hparams/arrange/3_reduction.yaml"
        config_fp = default_hparam_fp
        config = mlconfig.load(default_hparam_fp)
    else:
        config_fp = sys.argv[1]
        assert os.path.exists(config_fp), "Config file not found"
        config = mlconfig.load(config_fp)

    # Prepare MIDI file
    midi_fp_dict = read_yaml("../utils_arrange/song_path.yaml")
    assert config["song_name"] in midi_fp_dict, "Song not found in the song_path.yaml"
    midi_fp = midi_fp_dict[config["song_name"]]

    # Prepare paths
    exp_name = config["reinst_exp_name"]

    # Load the model and tokenizer
    tk_fp = config["tokenizer_fp"]
    tk = AutoTokenizer.from_pretrained(tk_fp)
    out_dir = jpath(config["result_root"], config["out_dir"])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    pt_ckpt = config["pt_ckpt"]
    model_cls = eval(config["lit_model_class"])
    l_model = model_cls.load_from_checkpoint(
        ckpt_fp, pt_ckpt=pt_ckpt, tokenizer=tk, infer=True
    )  # TODO: change to model_fp
    model = l_model.model
    model.eval()

    # Generate remi seqs, save to song_seq_remi.txt
    remi_tk = RemiTokenizer()
    midi_dir_name = os.path.dirname(midi_fp)
    remi_fp = jpath(midi_dir_name, "song_remi.txt")
    song_remi_seq = remi_tk.midi_to_remi(
        midi_fp,
        return_pitch_shift=False,
        return_key=False,
        normalize_pitch=False,
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
        if "i-128" in insts_with_voice:
            insts_with_voice.remove("i-128")
        for inst in insts_with_voice:

            # Quantize instrument
            inst_id = int(inst.split("-")[1])
            inst_id_quant = inst_util.slakh_quantize_inst_prog(inst_id)
            if inst_id_quant is None:
                inst_id_quant = inst_id
                # continue
            inst_tok = "i-{}".format(inst_id_quant)

            bar_seq_new.append(inst_tok)
            bar_seq_new.extend(opd_seqs[inst])
        bar_seq_new.append("b-1")

        remi_seq_new.extend(bar_seq_new)
    song_remi_seq = remi_seq_new
    RemiUtil.save_remi_seq(song_remi_seq, remi_fp)

    # detokenized_fp = jpath(midi_dir_name, "detokenized.mid")
    # remi_tk.remi_to_midi(song_remi_seq, detokenized_fp)
    seg_remi_seqs = remi_utils.song_remi_split_to_segments_2bar_hop1(
        song_remi_seq, ts_and_tempo=False
    )
    seg_remi_seqs_fp = jpath(midi_dir_name, "seg_remi.txt")
    t = [" ".join(i) + "\n" for i in seg_remi_seqs]
    with open(seg_remi_seqs_fp, "w") as f:
        f.writelines(t)

    # Construct dataset,
    bs = 1
    split = "infer"
    dataset_class = (
        config["dataset_class"] if "dataset_class" in config else "ArrangerDataset"
    )
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

    t = tk.pad_token

    # Config inference
    generate_kwargs = {
        # 'min_length': 500,
        "max_length": 800,
        "use_cache": True,
        "bad_words_ids": [
            [tk.pad_token_id],
            [tk.convert_tokens_to_ids("[PAD]")],
            [tk.convert_tokens_to_ids("[INST]")],
            [tk.convert_tokens_to_ids("[SEP]")],
        ],
        "no_repeat_ngram_size": (
            config["no_repeat_token"] if "no_repeat_token" in config else 6
        ),
        # # Greedy decode
        # 'do_sample': False,
        # 'num_beams': 5,
        # 'do_sample': False,
        # Sampling
        "do_sample": True,  # User searching method
        "top_k": config["top_k"],
        "top_p": config["top_p"],
        "temperature": config["temp"],
        # # Contrastive search
        # 'do_sample': False,
        # 'penalty_alpha': 0.6, #0.6
    }

    if "replace_inst" in config and config["replace_inst"] is not False:
        print("Inst setting: ", config["replace_inst"])

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
            tot_seq = batch[0].strip().split(" ")
            sep_idx = tot_seq.index("[SEP]")
            inp_seq = tot_seq[: sep_idx + 1]

            # Get the batch, replace the hist
            if config["hist_autoregressive"]:
                if hist_remi != None:

                    # Replace the hist subseq
                    hist_start_idx = inp_seq.index("[HIST]") + 1
                    hist_end_idx = inp_seq.index("[SEP]")
                    inp_seq = (
                        inp_seq[:hist_start_idx] + hist_remi + inp_seq[hist_end_idx:]
                    )

            # Replace the instrument
            if "replace_inst" in config and config["replace_inst"] is not False:
                inst_spec = config["replace_inst"]

                inst_start_idx = inp_seq.index("[INST]") + 1
                if "[MELODY]" not in inp_seq:
                    inst_end_idx = inp_seq.index("[PITCH]")
                else:
                    inst_end_idx = inp_seq.index("[MELODY]")
                inp_seq = inp_seq[:inst_start_idx] + inst_spec + inp_seq[inst_end_idx:]

            # Prepare input string
            inp_str = " ".join(inp_seq)
            inputs.append(inp_str)

            # Tokenize the batch
            batch_tokenized = tk(
                [inp_str],
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,  # Don't add eos token
            )["input_ids"].cuda()

            # Feed to the model
            out = model.generate(
                batch_tokenized, pad_token_id=tk.pad_token_id, **generate_kwargs
            )
            out_str = tk.batch_decode(out)[0]  # because we do bs=1 here

            # Select substr between [SEP] and [EOS] as output
            out_str = out_str.split("[SEP]")[1].split("[EOS]")[0].strip()

            # Truncate only useful part, i.e., between <sep> and </s>
            out_seq = out_str.split(" ")

            # Ensure the output part is a bar
            if out_seq[-1] != "b-1" and out_seq.count("b-1") == 0:
                out_seq.append("b-1")

            # Truncate content more than 1 bar
            bar_idx = out_seq.index("b-1")
            out_seq = out_seq[: bar_idx + 1]

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
    save_dir = jpath(out_dir, "lightning_logs", latest_version_dir, "infer")
    midi_out_dir = jpath(save_dir, "out_midi", config["reinst_group"])
    token_inp_dir = jpath(save_dir, "inp_token", config["reinst_group"])
    token_out_dir = jpath(save_dir, "out_token", config["reinst_group"])
    create_dir_if_not_exist(midi_out_dir)
    create_dir_if_not_exist(token_inp_dir)
    create_dir_if_not_exist(token_out_dir)

    # Save input tokens
    save_fn = "{}.txt".format(exp_name)
    input_save_fp = jpath(token_inp_dir, save_fn)
    with open(input_save_fp, "w") as f:
        f.writelines([i + "\n" for i in inputs])

    # Save output tokens
    save_fn = "{}.txt".format(exp_name)
    out_remi_fp = jpath(token_out_dir, save_fn)
    RemiUtil.save_remi_seq(test_out, out_remi_fp)

    # Save midi
    save_fn = "{}.mid".format(exp_name)
    out_midi_fp = jpath(midi_out_dir, save_fn)
    remi_tk.remi_to_midi(test_out, out_midi_fp)


if __name__ == "__main__":
    main()

"""
Create MIDI-only dataset from original Slakh dataset for MuseCoco fine-tuning
"""

import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../datasets")
sys.path.append("../utils_chord")

import shutil
import h5py
from tqdm import tqdm

from utils_midi.utils_midi import RemiUtil, RemiTokenizer



from src.utils import (
    save_json,
    read_json,
    create_dir_if_not_exist,
    accumulate_dic,
    print_json,
    sort_dict_by_key,
)
from src.utils import (
    get_hostname,
    jpath,
    ls,
    get_dataset_loc,
    get_dataset_dir,
    read_yaml,
    pexist,
    update_dic,
)
from src.utils import save_remi, read_remi, ChordUtil
from dataset_preparation.quantize_data import read_detected_chord
from utils_instrument.inst_map import InstMapUtil
from utils_chord.chord_map import ChordTokenizer
from utils_midi.utils_midi import RemiTokenizer
from utils_midi import remi_utils
from utils_texture.texture_tools import get_time_function_from_remi_one_bar, get_onset_density_of_a_bar_from_remi, tokenize_onset_density_one_bar
from utils_texture import texture_tools
import numpy as np

ls = os.listdir
jpath = os.path.join


def _main():
    # pre_process = MidiFeatureExtraction()
    # pre_process.process()

    pre_process = TwoBarDatasetPreparation()
    pre_process.process()


def transpose_chord(chord, pitch_shift):
    # 定义和弦根音到MIDI值的映射
    notes_to_midi = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    # MIDI值到和弦根音的映射
    midi_to_notes = {v: k for k, v in notes_to_midi.items()}

    # 如果和弦为"N"，直接返回
    if chord == "N":
        return "N"

    # 提取和弦的根音和类型
    root, chord_type = chord.split(":")

    # 计算新的根音
    if root in notes_to_midi:  # 确保根音有效
        original_midi = notes_to_midi[root]
        new_midi = (original_midi + pitch_shift) % 12  # 考虑循环
        new_root = midi_to_notes[new_midi]
    else:
        return "Invalid chord"  # 如果根音无效，返回错误

    # 返回转调后的和弦
    return f"{new_root}:{chord_type}"



def dataset_statistics():
    """
    Count the number of samples in different splits
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented/2-bar"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)
    for split in meta:
        split_entry = meta[split]
        print("{}, {}".format(split, len(split_entry)))


def obtain_time_function_from_remi():
    """
    Obtain an input feature, "time function", from the remi of each sample
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)
    tf_sum = [0 for i in range(48)]
    bar_cnt = 0
    for split in meta:
        split_entry = meta[split]
        pbar = tqdm(split_entry)
        for song in pbar:
            pbar.set_description(song)
            song_entry = split_entry[song]

            remi_fp = song_entry["remi_fp"]
            remi_seq = read_remi(remi_fp)
            b_1_indexes = [
                index for index, element in enumerate(remi_seq) if element == "b-1"
            ]
            b_1_indexes.insert(0, 0)

            tf_song = {}

            # Create the time function for each bar
            for bar_id in range(len(b_1_indexes) - 1):
                bar_start_idx = b_1_indexes[bar_id]
                next_bar_idx = b_1_indexes[bar_id + 1]
                bar_seq = remi_seq[
                    bar_start_idx:next_bar_idx
                ]  # 其实不太准，因为包含了b-1的位置，但能用

                tf = [0 for i in range(48)]  # number of onsets in each position
                pos = 0
                for tok in bar_seq:
                    if tok.startswith("o-"):
                        pos = int(tok.strip().split("-")[-1])
                    if tok.startswith("p-"):
                        tf[pos] += 1

                tf_str_list = ["TF-{}".format(i) for i in tf]
                tf_str = " ".join(tf_str_list)
                tf_song[bar_id] = tf_str
                # tf_song.extend(tf_str)

                bar_cnt += 1
                for i in range(len(tf_sum)):
                    tf_sum[i] += tf[i]

            # Get time function for a sample, save to file
            sample_dir = os.path.dirname(remi_fp)
            tf_fp = jpath(sample_dir, "time_func.txt")
            save_json(tf_song, tf_fp)
            # save_remi(tf_song, tf_fp)



def obtain_instrument_of_each_bar():
    """
    Obtain the instrument info of each bar.
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)

    inst_util = InstMapUtil()

    tf_sum = [0 for i in range(48)]
    bar_cnt = 0
    for split in meta:
        split_entry = meta[split]
        pbar = tqdm(split_entry)
        for song in pbar:
            pbar.set_description(song)
            song_entry = split_entry[song]

            remi_fp = song_entry["remi_fp"]
            remi_seq = read_remi(remi_fp)
            b_1_indexes = [
                index for index, element in enumerate(remi_seq) if element == "b-1"
            ]
            b_1_indexes.insert(0, 0)

            ins_song = {}

            # Detect the instrument for each bar
            for bar_id in range(len(b_1_indexes) - 1):
                bar_start_idx = b_1_indexes[bar_id]
                next_bar_idx = b_1_indexes[bar_id + 1]
                bar_seq = remi_seq[
                    bar_start_idx:next_bar_idx
                ]  # 其实不太准，因为包含了b-1的位置，但能用

                inst_bar = set()
                for tok in bar_seq:
                    if tok.startswith("i-"):
                        prog = int(tok.strip().split("-")[-1])
                        (
                            inst_id_str,
                            _,
                        ) = inst_util.slakh_from_midi_program_get_id_and_inst(prog)
                        inst_id = int(inst_id_str)
                        if inst_id not in inst_bar:
                            inst_bar.add(inst_id)

                inst_bar = list(inst_bar)
                inst_bar.sort()
                inst_bar_str_list = ["INS-{}".format(i) for i in inst_bar]
                inst_bar_str = " ".join(inst_bar_str_list)
                ins_song[bar_id] = inst_bar_str

                bar_cnt += 1

            # Get time function for a sample, save to file
            sample_dir = os.path.dirname(remi_fp)
            inst_fp = jpath(sample_dir, "inst_per_bar.txt")
            save_json(ins_song, inst_fp)


def normalize_chord_vocab():
    """
    Normalize chord vocab to Meganta's style
    """
    chord_util = ChordUtil()
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)

    tf_sum = [0 for i in range(48)]
    bar_cnt = 0
    for split in meta:
        split_entry = meta[split]
        pbar = tqdm(split_entry)
        for song in pbar:
            pbar.set_description(song)
            song_entry = split_entry[song]
            chords_str = song_entry["chords"]
            if len(chords_str) > 0:
                chord_seq = chords_str.strip().split(" ")
            else:
                chord_seq = ["N"]
            chord_seq_normalized = chord_util.normalize_chord_vocab_for_chord_seq(
                chord_seq
            )
            chord_seq_normalized_str = " ".join(chord_seq_normalized)
            meta[split][song]["chords_meganta"] = chord_seq_normalized_str
    save_json(meta, meta_path)


def tokenize_conditions():
    """
    Put all information of conditions in a single list of tokens
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)
    chord_tk = ChordTokenizer()

    for split in meta:
        split_entry = meta[split]
        pbar = tqdm(split_entry)
        for song in pbar:
            pbar.set_description(song)
            song_entry = split_entry[song]

            """
            Input sequence format:
            (info of entire sample)
                (key info)                          K-0
            (info of first bar) 
                (inst info)                         INS  INS-0  INS-3   (id small to large)
                    (info of first beat)
                        (chord)                     CD CR-0 CT-1 
                        (time function)             TF TF-8  TF-0  TF-3 (12 TF tokens) 
                    ...
                    (info of rest 3 beats)  
                (bar line)                          b-1 
            ...
            (conditions of the rest 7 bars)
            """
            tokens = []

            # Remi
            remi_fp = song_entry["remi_fp"]
            remi_seq = read_remi(remi_fp)
            b_1_indexes = [
                index for index, element in enumerate(remi_seq) if element == "b-1"
            ]
            num_bars = len(b_1_indexes)
            b_1_indexes.insert(0, -1)

            # Key
            key = song_entry["key"]  # major | minor

            # Instrument
            sample_dir = os.path.dirname(remi_fp)
            # print(sample_dir)
            inst_fp = jpath(sample_dir, "inst_per_bar.txt")
            inst_per_bar = read_json(inst_fp)

            # Chord
            chords = song_entry["chords"].strip().split(" ")
            # print(chords)
            chords_per_bar = 4
            # chords_tokenized = chord_tk.tokenize_chord_list(chords)

            # Time Function
            time_func_fp = jpath(sample_dir, "time_func.txt")
            time_func = read_json(time_func_fp)

            # Collate
            # Add Key
            if key == "major":
                key_token = "K-0"
            else:
                key_token = "K-1"
            tokens.append(key_token)

            # print(num_bars)

            # Iterate over all bars
            for bar_id in range(num_bars):
                bar_start_idx = b_1_indexes[bar_id] + 1
                bar_end_idx = b_1_indexes[bar_id + 1]
                bar_seq = remi_seq[bar_start_idx:bar_end_idx]

                # Add Inst
                tokens.append("INS")
                bar_id_str = str(bar_id)
                inst_this_bar = inst_per_bar[bar_id_str]
                if len(inst_this_bar) > 0:
                    inst_this_bar = inst_this_bar.split(" ")
                    tokens.extend(inst_this_bar)

                # Obtain chord and tine func
                chord_idx = bar_id * chords_per_bar
                chord_this_bar = chords[chord_idx : chord_idx + chords_per_bar]

                if len(chord_this_bar) < 1:
                    chord_this_bar.append(
                        "N:N"
                    )  # Ensure each bar have at lease one chord symbol
                if len(chord_this_bar[0]) == 0:
                    chord_this_bar[0] = "N:N"

                # print(chord_this_bar, len(chord_this_bar))
                chord_tokenized = chord_tk.tokenize_chord_list(chord_this_bar)
                time_func_this_bar = time_func[bar_id_str].split(" ")

                for beat_id in range(4):  # for each beat, id 0~3
                    # First bar: chord
                    tokens.append("CD")
                    chord_tokens_per_beat = 2
                    tokens.extend(
                        chord_tokenized[
                            beat_id
                            * chord_tokens_per_beat : beat_id
                            * chord_tokens_per_beat
                            + chord_tokens_per_beat
                        ]
                    )

                    # First bar: time function
                    tfs_per_beat = 12
                    time_func_1st_half = time_func_this_bar[
                        beat_id * tfs_per_beat : beat_id * tfs_per_beat + tfs_per_beat
                    ]
                    tokens.append("TF")
                    tokens.extend(time_func_1st_half)

                # Add bar line
                tokens.append("b-1")

            tokens_str = " ".join(tokens)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_condition.txt")
            meta[split][song]["tokenized_condition_fp"] = tokenized_condition_fp
            with open(tokenized_condition_fp, "w") as f:
                f.write(tokens_str + "\n")

            # exit(10)

    save_json(meta, meta_path)


def collate_into_single_files():
    """
    Collate all input and output from all samples into a single file.
    Note: won't include samples that exceed the upper length limit
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)
    chord_tk = ChordTokenizer()
    out_dir = jpath(data_dir, "collated")
    create_dir_if_not_exist(out_dir)

    length_limit = 3190

    for split in meta:
        split_entry = meta[split]
        if split == "validation":
            split_name = "valid"
        else:
            split_name = split
        pbar = tqdm(split_entry)

        src_data = []
        tgt_data = []
        for song in pbar:
            pbar.set_description(song)
            song_entry = split_entry[song]
            condition_fp = song_entry["tokenized_condition_fp"]
            remi_fp = song_entry["remi_fp"]
            remi_song = read_remi(remi_fp, split=False)
            condition_song = read_remi(condition_fp, split=False)

            # Length check
            remi_seq = remi_song.split(" ")
            input_seq = condition_song.split(" ")
            if len(remi_seq) + len(input_seq) > length_limit:
                continue

            tgt_data.append(remi_song + "\n")
            src_data.append(condition_song + "\n")

        split_out_fp = jpath(out_dir, "{}.txt".format(split_name))
        with open(split_out_fp, "w") as f:
            f.writelines(tgt_data)

        split_out_fp = jpath(out_dir, "{}_input.txt".format(split_name))
        with open(split_out_fp, "w") as f:
            f.writelines(src_data)


def check_input_tokens():
    """
    Take of look about the input tokens
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented"
    meta_path = jpath(data_dir, "metadata.json")
    meta = read_json(meta_path)
    chord_tk = ChordTokenizer()
    out_dir = jpath(data_dir, "collated")
    tokens = {}
    for split in ["test", "valid", "train"]:
        input_fp = jpath(out_dir, "{}_input.txt".format(split))
        with open(input_fp) as f:
            data = f.readlines()
        for line in data:
            line = line.strip()
            token_seq = line.split(" ")
            for token in token_seq:
                if token not in tokens:
                    tokens[token] = 1
                else:
                    tokens[token] += 1

    out_fp = jpath(out_dir, "input_tokens.json")
    token_list = list(tokens.items())
    token_list.sort()
    tokens = dict(token_list)
    save_json(tokens, out_fp)


def observe_sequence_length():
    """
    Check the length of sequence in both src and tgt data.
    """
    data_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented/collated"
    out_dir = jpath(data_dir, "statistics")
    create_dir_if_not_exist(out_dir)
    sample_cnt = {}

    for split in ["train", "valid", "test"]:
        tot_token_cnt = []
        src_fn = split + "_input.txt"  # 重新统计长度
        src_fp = jpath(data_dir, src_fn)
        tgt_fn = "{}.txt".format(split)
        tgt_fp = jpath(data_dir, tgt_fn)
        with open(src_fp) as f:
            src_data = f.readlines()
        with open(tgt_fp) as f:
            tgt_data = f.readlines()
        tot_seq_len_list = []
        src_seq_len_list = []
        for src_sent, tgt_sent in zip(src_data, tgt_data):
            src_seq = src_sent.strip().split(" ")
            tgt_seq = tgt_sent.strip().split(" ")
            tot_len = len(src_seq) + len(tgt_seq)
            tot_seq_len_list.append(tot_len)
            src_seq_len_list.append(len(src_seq))

        # for side in ['src', 'tgt']:
        #     token_cnt = {}
        #
        #     if side == 'src':
        #         data_fn = split + '_input.txt'
        #     else:
        #         data_fn = split + '.txt'
        #     data_fp = jpath(data_dir, data_fn)
        #     with open(data_fp) as f:
        #         data = f.readlines()
        #     num_samples = len(data)
        #
        #
        #     for sample in data:
        #         tokens = sample.strip().split(' ')
        #         num_token = len(tokens)
        #         accumulate_dic(token_cnt, num_token)
        #         tot_seq_len_list.append(num_token)
        #
        #     token_cnt = sort_dict_by_key(token_cnt, reverse=True)
        #     # print_json(token_cnt)
        #     out_fp = jpath(out_dir, 'token_cnt_{}-{}.json'.format(split, side))
        #     save_json(token_cnt, out_fp)
        #
        #     tot_token_cnt.extend(tot_seq_len_list)
        #     # t = np.quantile(num_token_list, 1)
        #     # print(t)

        # sample_cnt['{}'.format(split)] = num_samples
        t = np.quantile(src_seq_len_list, 0.1)
        print("{}: {}".format(split, t))

    out_fp = jpath(out_dir, "sample_cnt.json")
    save_json(sample_cnt, out_fp)


class MidiFeatureExtraction:
    """
    Extract features from MIDI to prepare training data.
    """

    def __init__(self):
        self.data_dir = get_dataset_dir()
        self.meta_path = jpath(self.data_dir, "metadata.json")

    def process(self):
        # self.create_meta()
        # self.obtain_token_sequence()
        # self.modify_remi()
        # self.get_bar_positions()
        # self.get_chord_seq()
        # self.normalize_chords()
        # self.obtain_key()
        # self.get_bar_positions()
        self.segment_data()

        # dataset_statistics()

    def create_meta(self):
        """
        Create metadata with keys in hdf5
        :return:
        """
        dataset_path = self.data_dir
        metadata_fp = jpath(dataset_path, "metadata.json")
        meta = {}

        h5_fp = get_dataset_loc()
        h5_data = h5py.File(h5_fp, "r")
        pbar = tqdm(h5_data)
        for song_name in pbar:
            pbar.set_description(song_name)
            song_entry = h5_data[song_name]
            split_name = song_entry.attrs["split"]
            meta[song_name] = {"split": split_name}

        save_json(meta, metadata_fp)

    def obtain_token_sequence(self):
        """
        Obtain REMI-like token sequence for all songs, save to dir of each song.
        """
        dataset_path = self.data_dir

        tk = RemiTokenizer()

        splits = ["test", "validation", "train"]
        for split in splits:
            split_dir = jpath(dataset_path, split)

            songs = ls(split_dir)
            songs.sort()
            pbar = tqdm(songs)
            for song in pbar:
                pbar.set_description(song)
                if song.startswith("."):
                    continue

                song_dir = jpath(split_dir, song)
                midi_fp = jpath(song_dir, "all_src.mid")
                remi_fp = jpath(song_dir, "remi.txt")

                pitch_shift = tk.midi_to_remi_file(
                    midi_fp, remi_fp, return_pitch_shift=True
                )
                pitch_shift_fp = jpath(song_dir, "pitch_shift")
                with open(pitch_shift_fp, "w") as f:
                    f.write("{}".format(pitch_shift))

    def get_bar_positions(self):
        dataset_path = self.data_dir

        tk = RemiTokenizer()

        splits = ["test", "validation", "train"]
        for split in splits:
            split_dir = jpath(dataset_path, split)

            songs = ls(split_dir)
            songs.sort()
            pbar = tqdm(songs)
            for song in pbar:
                pbar.set_description(song)
                if song.startswith("."):
                    continue

                song_dir = jpath(split_dir, song)
                remi_fp = jpath(song_dir, "remi.txt")
                with open(remi_fp) as f:
                    remi_str = f.readline()
                # print(len(remi_str), remi_str[-10:])
                remi_seq = remi_str.strip().split(" ")
                bar_start_token_indices = tk.get_bar_pos(remi_seq)

                bar_fp = jpath(song_dir, "bar_start_token_indices.json")
                save_json(bar_start_token_indices, bar_fp)

    def get_chord_seq(self):
        """
        Bar id * 2 is chord id.
        :return:
        """
        print("Obtaining quantized chord sequence from chord detection results ...")

        meta = read_json(self.meta_path)
        for song_name in tqdm(meta):
            song_entry = meta[song_name]
            split = song_entry["split"]
            song_dir = jpath(self.data_dir, split, song_name)
            assert os.path.exists(song_dir)

            # prepare pos_in_sec
            h5_fp = get_dataset_loc()
            h5_data = h5py.File(h5_fp, "r")
            song_entry_h5 = h5_data[song_name]
            pos_of_16th_note = song_entry_h5["pos_in_sec"][()]

            detected_chord_path = jpath(song_dir, "detect_chords_from_midi.txt")
            assert os.path.exists(detected_chord_path)
            chord_tuple_list = read_detected_chord(detected_chord_path)

            chord_seq = quantize_chord(
                detected_chords=chord_tuple_list,
                pos_16th=pos_of_16th_note,
                num_beat_per_chord=2,
            )
            # chord_seq = [i.decode("utf-8") for i in chord_seq]
            chord_str = " ".join(chord_seq)

            # song_dir = jpath(dataset_path, split_name, song_name)
            chord_fp = jpath(song_dir, "chords_per_half_bar.txt")
            with open(chord_fp, "w") as f:
                f.write(chord_str + "\n")

    def normalize_chords(self):
        """
        Normalize the root note of chord according to pitch_shift
        Add "chord_normalized" to metadata
        """
        dataset_path = self.data_dir
        meta_path = jpath(dataset_path, "metadata.json")
        meta = read_json(meta_path)

        pbar = tqdm(meta)
        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            song_dir = jpath(dataset_path, split, song)
            chord_fp = jpath(song_dir, "chords_per_half_bar.txt")
            with open(chord_fp) as f:
                chords = f.readline()
            chords = chords.strip()

            pitch_shift_fp = jpath(song_dir, "pitch_shift")
            with open(pitch_shift_fp) as f:
                t = f.read().strip()
            pitch_shift = int(t)

            chords = chords.split(" ")
            normalized_chord_list = []
            for chord in chords:
                chord_normalized = transpose_chord(chord, pitch_shift)
                normalized_chord_list.append(chord_normalized)

            normalized_chords = " ".join(normalized_chord_list)
            meta[song]["chords_normalized"] = normalized_chords

        save_json(meta, meta_path)

    def obtain_key(self):
        """
        Obtain the key during tokenization of midi, save such info in metadata
        """
        dataset_path = self.data_dir
        meta_path = self.meta_path
        meta = read_json(meta_path)

        tk = RemiTokenizer()

        pbar = tqdm(meta)
        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            song_dir = jpath(dataset_path, split, song)
            midi_fp = jpath(song_dir, "all_src.mid")

            remi, pitch_shift, is_major = tk.midi_to_remi(
                midi_fp, return_pitch_shift=True, return_key=True
            )
            assert is_major in [True, False]
            if is_major:
                key = "major"
            else:
                key = "minor"
            meta[song]["key"] = key

        save_json(meta, meta_path)

    def modify_remi(self):
        """
        Modify remi after tokenization
        - Remove notes from instruments that are not supported by Slakh
        - Quantize instrument token
        """
        data_dir = self.data_dir
        meta_path = self.meta_path
        meta = read_json(meta_path)
        inst_util = InstMapUtil()

        # for split in meta:
        pbar = tqdm(meta)
        for song in pbar:
            pbar.set_description(song)

            song_entry = meta[song]
            split = song_entry["split"]
            song_dir = jpath(data_dir, split, song)
            remi_fp = jpath(song_dir, "remi.txt")
            shutil.copy(remi_fp, remi_fp.replace(".txt", "raw.txt"))

            remi_seq = read_remi(remi_fp)

            # Remove not support instruments
            # 从后往前遍历列表，这样在删除元素时不会干扰到索引
            i = len(remi_seq) - 1
            while i >= 0:
                remi_tok = remi_seq[i]
                prog_ori = remi_tok.split("-")[-1]
                # 检查当前元素是否满足条件A
                if not inst_util.slack_support_instrument(
                    prog_ori
                ):  # 请替换“满足条件A”为具体的条件判断
                    # 确保当前元素以'i-'开头，且存在两个后续元素分别以'p-'和'd-'开头
                    if (
                        remi_seq[i].startswith("i-")
                        and i + 1 < len(remi_seq)
                        and remi_seq[i + 1].startswith("p-")
                        and i + 2 < len(remi_seq)
                        and remi_seq[i + 2].startswith("d-")
                    ):
                        # 删除满足条件的元素及其后面的两个元素
                        del remi_seq[i : i + 3]
                        # 调整索引，跳过已删除的元素
                        i -= 3
                        continue
                i -= 1
            len_final = len(remi_seq)

            # Quantize instrument info
            remi_seq_new = []
            for remi_tok in remi_seq:
                if remi_tok.startswith("i-"):
                    inst_old = remi_tok
                    prog_ori = inst_old.split("-")[-1]
                    assert inst_util.slack_support_instrument(prog_ori)
                    prog_new = inst_util.slakh_quantize_inst_prog(prog_ori)
                    inst_new = "i-{}".format(prog_new)
                    remi_seq_new.append(inst_new)
                else:
                    remi_seq_new.append(remi_tok)

            # Save the modified remi file
            remi_out_fp = remi_fp
            save_remi(remi_seq_new, remi_out_fp)

            # Modify remi entry in metadata
            # meta[song]['remi_fp'] = remi_out_fp

        # save_json(meta, meta_path)

    def segment_data_old(self):
        """
        Segment the entire dataset. Segment each remi to 8-bar samples, with 4-bar overlap.
        """
        dataset_path = self.data_dir
        meta_path = jpath(dataset_path, "metadata.json")
        meta = read_json(meta_path)

        out_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented/2-bar-strict"
        create_dir_if_not_exist(out_dir)
        segment_data_dir = jpath(out_dir, "data")
        create_dir_if_not_exist(segment_data_dir)

        meta_seg_fp = jpath(out_dir, "metadata.json")
        meta_seg = {}

        pbar = tqdm(meta)
        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            # Get the song-level remi content
            song_dir = jpath(dataset_path, split, song)
            remi_fp = jpath(song_dir, "remi.txt")
            with open(remi_fp) as f:
                remi_str = f.readline()
            remi_seq = remi_str.strip().split(" ")

            # Get the bar positions (remi already midified)
            bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
            num_bars = len(bar_indices)
            # bar_start_idx_fp = jpath(song_dir, "bar_start_token_indices.json")
            # bar_start_idx = read_json(bar_start_idx_fp)

            # Define the sample length here
            bars_per_sample = 2
            hop_bars = 1

            # NOTE: The first sample need to contain an empty bar and the first bar
            # num_segment = num_bar. 
            # bar_start_idx.insert(0, (0,0))
            # a = 1

            # Start segmentation, for all bar ids
            num_bars = len(bar_indices)
            max_bar_id = num_bars - 1
            segment_id = 0
            for tgt_bar_id in bar_indices:
                # Get idx of the second bar
                tgt_bar_start_idx, tgt_bar_end_idx = bar_indices[tgt_bar_id]

                # Obtain content of the first bar
                if tgt_bar_id > 0: # For second and later bars
                    hist_bar_id = tgt_bar_id - 1
                    hist_bar_start_idx, hist_bar_end_idx = bar_indices[hist_bar_id]
                    hist_bar_tokens = remi_seq[hist_bar_start_idx:hist_bar_end_idx]
                else: # For the first bar
                    # Create an empty bar
                    hist_bar_tokens = ['b-1']
                
                #
                
            for start_bar_id in range(0, num_bars, hop_bars):
                end_bar_id = min(max_bar_id, start_bar_id + bars_per_sample - 1)
                start_bar_token_idx = bar_indices[str(start_bar_id)][0]
                end_bar_token_idx = bar_indices[str(end_bar_id)][1]
                # the starting idx of next sample. actual ending token idx + 1.

                # Create sample dir in segmented dataset

                segment_remi_seq = remi_seq[start_bar_token_idx:end_bar_token_idx]
                segment_remi = " ".join(segment_remi_seq)

                # Save the segment remi
                segment_dir_name = "{}-{}".format(song, segment_id)
                segment_dir = jpath(segment_data_dir, segment_dir_name)
                create_dir_if_not_exist(segment_dir)
                seg_remi_fp = jpath(segment_dir, "remi.txt")
                with open(seg_remi_fp, "w") as f:
                    f.write(segment_remi)

                # Segment chords
                start_chord_id = start_bar_id * 2
                chords_seq = meta[song]["chords_normalized"].strip().split(" ")
                segmented_chords = chords_seq[
                    start_chord_id : start_chord_id + bars_per_sample * 2
                ]
                segmented_chords = " ".join(segmented_chords)

                meta_seg[segment_dir_name] = {
                    "split": split,
                    "key": meta[song]["key"],
                    "chords": segmented_chords,
                    "remi": seg_remi_fp,
                }

                segment_id += 1

        save_json(meta_seg, meta_seg_fp)

    def segment_data(self):
        """
        Segment the entire dataset. Segment each remi to 8-bar samples, with 4-bar overlap.
        """
        dataset_path = self.data_dir
        meta_path = jpath(dataset_path, "metadata.json")
        meta = read_json(meta_path)

        out_dir = "/data2/longshen/Datasets/slakh2100_flac_redux/segmented/2-bar-strict"
        create_dir_if_not_exist(out_dir)
        segment_data_dir = jpath(out_dir, "data")
        create_dir_if_not_exist(segment_data_dir)

        meta_seg_fp = jpath(out_dir, "metadata.json")
        meta_seg = {}

        pbar = tqdm(meta)
        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            # Get the song-level remi content
            song_dir = jpath(dataset_path, split, song)
            remi_fp = jpath(song_dir, "remi.txt")
            with open(remi_fp) as f:
                remi_str = f.readline()
            remi_seq = remi_str.strip().split(" ")

            # Get the bar positions (remi already midified)
            bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)

            # NOTE: The first sample need to contain an empty bar and the first bar
            # Start segmentation, for all bar ids
            # 2 bars in each segment.
            segment_id = 0
            for tgt_bar_id in bar_indices:
                # Get idx of the second bar
                tgt_bar_start_idx, tgt_bar_end_idx = bar_indices[tgt_bar_id]
                tgt_bar_tokens = remi_seq[tgt_bar_start_idx:tgt_bar_end_idx]

                # Obtain content of the first bar
                if tgt_bar_id > 0: # For second and later bars
                    hist_bar_id = tgt_bar_id - 1
                    hist_bar_start_idx, hist_bar_end_idx = bar_indices[hist_bar_id]
                    hist_bar_tokens = remi_seq[hist_bar_start_idx:hist_bar_end_idx]
                else: # For the first bar
                    # Create an empty bar
                    hist_bar_tokens = ['b-1']
                
                # Obtain the segment content by concate the two bars
                segment_remi_seq = hist_bar_tokens
                segment_remi_seq.extend(tgt_bar_tokens)
                segment_remi_str = " ".join(segment_remi_seq)

                # Save the segment remi
                segment_dir_name = "{}-{}".format(song, segment_id)
                segment_dir = jpath(segment_data_dir, segment_dir_name)
                create_dir_if_not_exist(segment_dir)
                seg_remi_fp = jpath(segment_dir, "remi.txt")
                with open(seg_remi_fp, "w") as f:
                    f.write(segment_remi_str)

                meta_seg[segment_dir_name] = {
                    "split": split,
                    # "key": meta[song]["key"],
                    "remi": seg_remi_fp,
                }

                segment_id += 1

        save_json(meta_seg, meta_seg_fp)


class TwoBarDatasetPreparation:
    def __init__(self):
        self.song_level_meta_fp = (
            "/data2/longshen/Datasets/slakh2100_flac_redux/metadata.json"
        )
        self.output_dir = (
            "/data2/longshen/Datasets/slakh2100_flac_redux/segmented/2-bar-strict"
        )
        self.meta_before_resplit_fp = jpath(self.output_dir, "metadata.json")
        self.meta_fp = jpath(self.output_dir, "metadata_resplit.json")
        self.re_split_idx_fp = jpath(self.output_dir, "resplit_idx.json")

    def process(self):
        # self.obtain_idx_for_re_split()                # Re-split the dataset
        # self.re_split()                               # Physically split the dataset according to the new splitting
        # self.statistics()                             # Get dataset statistics

        ''' Choose an option to do the tokenization '''
        # self.tokenize_conditions_key_ins_chd()        # Tokenize the split dataset for conditional generation task
        # self.tokenize_conditions_key_ins_chd_txt()
        self.tokenize_for_source_separation()
        # self.tokenize_for_sss_inst_pitch_pos()
        # self.tokenize_for_sss_ipo_sort_pitch()
        # self.tokenize_for_sss_ipo_with_onset_count()
        self.tokenize_for_sss_ipo_tf_with_history()

        # self.collate_into_single_files()   # Set a large length limit for the first time
        # self.observe_sequence_length(quantile=1.0)
        # self.observe_sequence_length(quantile=0.5)
        # self.observe_sequence_length(quantile=0.95)
        # self.collate_into_single_files(
        #     len_limit=850
        # )  # Set a the length limit to a proper value
        
        # Below procedures are optional
        # self.de_tokenize()
        # self.observe_num_inst(0.75) # [0, 5, 6, 7, 12] # quantiles of num_inst

        # Chord rec debug
        # self.obtain_chord_from_segment()
        # self.compare_chord_difference()

    def obtain_idx_for_re_split(self):
        """
        Re-do the splitting for the dataset so that valid and testing split contains fewer samples.
        This function select id of songs to be remaining in validation and test set.
        """
        # Obtain song ids in valid and test set
        song_level_meta = self.meta_before_resplit_fp
        meta = read_json(song_level_meta)
        valid_ids = []
        test_ids = []

        # for split_name in meta:
        #     split = meta[split_name]
        #     for song_name in split:
        #         if split_name == 'validation':
        #             valid_ids.append(song_name)
        #         elif split_name == 'test':
        #             test_ids.append(song_name)
        for song_name in meta:
            song_entry = meta[song_name]
            if song_entry["split"] == "validation":
                valid_ids.append(song_name)
            elif song_entry["split"] == "test":
                test_ids.append(song_name)
        print(len(valid_ids), len(test_ids))

        # Random select a subset of ids from valid (1/11) and test (1/6)
        valid_select_ratio = 1 / 11
        test_select_ratio = 1 / 6
        import random

        valid_ids_new = random.sample(
            valid_ids, int(len(valid_ids) * valid_select_ratio)
        )
        test_ids_new = random.sample(test_ids, int(len(test_ids) * test_select_ratio))
        print(len(valid_ids_new), len(test_ids_new))

        indexes = {
            "valid": valid_ids_new,
            "test": test_ids_new,
        }
        indices_fp = jpath(self.output_dir, "resplit_idx.json")
        save_json(indexes, indices_fp)

    def re_split(self):
        """
        Re-do the splitting for the dataset so that valid and testing split contains fewer samples.
        Procedures:
        - Read the re_split index obtained previously
        - Restructure the metadata in the output_dir
        """
        re_split_idx = read_json(self.re_split_idx_fp)
        valid_idx = set(re_split_idx["valid"])
        test_idx = set(re_split_idx["test"])
        meta = read_json(self.meta_before_resplit_fp)
        meta_new = {}

        for segment_id in meta:
            # song_id = segment_id.strip().split('-')[0]
            if segment_id in valid_idx:
                meta[segment_id]["split"] = "validation"
                meta_new[segment_id] = meta[segment_id]
            elif segment_id in test_idx:
                meta[segment_id]["split"] = "test"
                meta_new[segment_id] = meta[segment_id]
            elif meta[segment_id]["split"] == "train":
                meta_new[segment_id] = meta[segment_id]
        #
        # train_split = meta['train']
        # for split in ['validation', 'test']:
        #     split_old = meta[split]
        #     split_new = {}
        #     for segment_id in meta[split]:
        #         song_id = segment_id.strip().split('-')[0]
        #         if song_id not in valid_idx and song_id not in test_idx:
        #             train_split[segment_id] = meta[split][segment_id]
        #         else:
        #             split_new[segment_id] = meta[split][segment_id]
        #     meta_new[split] = split_new
        # meta_new['train'] = train_split
        save_json(meta_new, self.meta_fp)

    def statistics(self):
        meta_path = jpath(self.output_dir, "metadata_resplit.json")
        meta = read_json(meta_path)
        res = {}
        for song in meta:
            split = meta[song]["split"]
            update_dic(res, split, song)
        for k in res:
            print(k, len(res[k]))
            # print('{}, {}'.format(song, len(split_entry)))

    def tokenize_conditions_ins_chd(self):
        """
        Put all information of conditions in a same string
        Generate the input sequence from remi sequence and conditions

        In this version, input sequence contains:
        - Instrument
        - Chord (2 chords each bar)
        """
        data_dir = self.output_dir
        meta_path = jpath(data_dir, "metadata_resplit.json")
        meta = read_json(meta_path)
        chord_tk = ChordTokenizer()

        pbar = tqdm(meta)
        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            """
            Input sequence format:
            (info of entire sample)
                (key info)                          K-0
            (info of first bar) 
                (inst info)                         INS  INS-0  INS-3   (id small to large)
                    (info of first beat)
                        (chord)                     CD CR-0 CT-1 
                        (time function)             TF TF-8  TF-0  TF-3 (12 TF tokens) 
                    ...
                    (info of rest 3 beats)  
                (bar line)                          b-1 
            ...
            (conditions of the rest 7 bars)
            """
            input_tokens = []

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)
            b_1_indexes = [
                index for index, element in enumerate(remi_seq) if element == "b-1"
            ]
            num_bars = len(b_1_indexes)
            b_1_indexes.insert(
                0, -1
            )  # to facilitate access first token of each bar, add a b-1 at the very beginning

            # # Key
            # key = song_entry['key']  # major | minor

            # # Instrument
            sample_dir = os.path.dirname(remi_fp)
            # # print(sample_dir)
            # inst_fp = jpath(sample_dir, 'inst_per_bar.txt')
            # inst_per_bar = read_json(inst_fp)

            # Chord
            chords_per_bar = 2
            song_dir = os.path.dirname(remi_fp)
            chord_seg_fp = jpath(song_dir, "chord_from_recon.txt")
            with open(chord_seg_fp) as f:
                chords = f.read().strip().split(" ")
            # chords = song_entry['chords'].strip().split(' ')

            # chords_tokenized = chord_tk.tokenize_chord_list(chords)

            # # Time Function
            # time_func_fp = jpath(sample_dir, 'time_func.txt')
            # time_func = read_json(time_func_fp)

            # # Collate
            # # Add Key
            # if key == 'major':
            #     key_token = 'K-0'
            # else:
            #     key_token = 'K-1'
            # input_tokens.append(key_token)

            # print(num_bars)

            # Iterate over all bars
            for bar_id in range(num_bars):
                bar_start_idx = b_1_indexes[bar_id] + 1
                bar_end_idx = b_1_indexes[bar_id + 1]
                bar_seq = remi_seq[bar_start_idx:bar_end_idx]

                """ Add Inst """
                # Obtain instruments from bar_seq
                insts_this_bar = set()
                for token in bar_seq:
                    if token.startswith("i-"):
                        insts_this_bar.add(token)
                insts_this_bar = list(insts_this_bar)
                insts_this_bar = sorted(
                    insts_this_bar, key=lambda x: int(x.split("-")[1])
                )  # sort by inst id

                # Add instrument info to input_tokens
                input_tokens.append("INS")
                input_tokens.extend(insts_this_bar)

                # Obtain chord and time func
                chord_idx = bar_id * chords_per_bar
                chord_this_bar = chords[chord_idx : chord_idx + chords_per_bar]

                if len(chord_this_bar) < 1:
                    chord_this_bar.append(
                        "N:N"
                    )  # Ensure each bar have at lease one chord symbol
                if len(chord_this_bar[0]) == 0:
                    chord_this_bar[0] = "N:N"

                # print(chord_this_bar, len(chord_this_bar))
                chord_tokenized = chord_tk.tokenize_chord_list(chord_this_bar)
                # time_func_this_bar = time_func[bar_id_str].split(' ')

                for beat_id in range(2):  # for each beat, id 0~3
                    # First bar: chord
                    input_tokens.append("CD")
                    chord_tokens_per_beat = 2
                    input_tokens.extend(
                        chord_tokenized[
                            beat_id
                            * chord_tokens_per_beat : beat_id
                            * chord_tokens_per_beat
                            + chord_tokens_per_beat
                        ]
                    )

                    # # First bar: time function
                    # tfs_per_beat = 12
                    # time_func_1st_half = time_func_this_bar[
                    #                      beat_id * tfs_per_beat:beat_id * tfs_per_beat + tfs_per_beat]
                    # input_tokens.append('TF')
                    # input_tokens.extend(time_func_1st_half)

                # Add bar line
                input_tokens.append("b-1")

            tokens_str = " ".join(input_tokens)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_condition.txt")
            meta[song]["tokenized_condition"] = tokens_str
            with open(tokenized_condition_fp, "w") as f:
                f.write(tokens_str + "\n")

            # exit(10)

        save_json(meta, meta_path)

    def tokenize_conditions_key_ins_chd(self):
        """
        Put all information of conditions in a same string
        Generate the input sequence from remi sequence and conditions

        In this version, input sequence contains:
        - Instrument
        - Key (major or minor)
        - Chord (2 chords each bar)
        """
        data_dir = self.output_dir
        meta_path = jpath(data_dir, "metadata_resplit.json")
        meta = read_json(meta_path)
        chord_tk = ChordTokenizer()

        pbar = tqdm(meta)
        # for split in meta:
        #     split_entry = meta[split]

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            input_tokens = []

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)
            b_1_indexes = [
                index for index, element in enumerate(remi_seq) if element == "b-1"
            ]
            num_bars = len(b_1_indexes)
            # To facilitate access first token of each bar, add a b-1 at the very beginning
            b_1_indexes.insert(0, -1)
            sample_dir = os.path.dirname(remi_fp)

            if num_bars == 0:
                raise Exception("Bar num = 0")

            # Key
            key = song_entry["key"]  # major | minor

            # Chord
            chords_per_bar = 2
            song_dir = os.path.dirname(remi_fp)
            chord_seg_fp = jpath(song_dir, "chord_from_recon.txt")
            with open(chord_seg_fp) as f:
                chords = f.read().strip().split(" ")

            # Collate
            # Add Key
            if key == "major":
                key_token = "K-0"
            else:
                key_token = "K-1"
            input_tokens.append(key_token)

            # Add scale
            scale_token = "S-1"
            input_tokens.append(scale_token)

            # Iterate over all bars
            for bar_id in range(num_bars):
                bar_start_idx = b_1_indexes[bar_id] + 1
                bar_end_idx = b_1_indexes[bar_id + 1]
                bar_seq = remi_seq[bar_start_idx:bar_end_idx]

                """ Add Inst """
                # Obtain instruments from bar_seq
                insts_this_bar = set()
                for token in bar_seq:
                    if token.startswith("i-"):
                        insts_this_bar.add(token)
                insts_this_bar = list(insts_this_bar)
                insts_this_bar = sorted(
                    insts_this_bar, key=lambda x: int(x.split("-")[1])
                )  # sort by inst id

                # Add instrument info to input_tokens
                input_tokens.append("INS")
                input_tokens.extend(insts_this_bar)

                # Obtain chord and time func
                chord_idx = bar_id * chords_per_bar
                chord_this_bar = chords[chord_idx : chord_idx + chords_per_bar]

                if len(chord_this_bar) < 1:
                    chord_this_bar.append(
                        "N:N"
                    )  # Ensure each bar have at lease one chord symbol
                if len(chord_this_bar[0]) == 0:
                    chord_this_bar[0] = "N:N"

                chord_tokenized = chord_tk.tokenize_chord_list(chord_this_bar)

                for beat_id in range(2):  # for each beat, id 0~3
                    # Chord tokens
                    input_tokens.append("CD")
                    chord_tokens_per_beat = 2
                    input_tokens.extend(
                        chord_tokenized[
                            beat_id
                            * chord_tokens_per_beat : beat_id
                            * chord_tokens_per_beat
                            + chord_tokens_per_beat
                        ]
                    )

                # Add bar line
                input_tokens.append("b-1")

            input_tokens_str = " ".join(input_tokens)
            tokenized_condition_fp = jpath(
                sample_dir, "tokenized_condition_key_inst_chd.txt"
            )
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            meta[song]["tokenized_condition"] = input_tokens_str

        save_json(meta, meta_path)

    def tokenize_conditions_key_ins_chd_txt(self):
        """
        Put all information of conditions in a same string
        Generate the input sequence from remi sequence and conditions

        In this version, input sequence contains:
        - Instrument
        - Key (major or minor)
        - Chord (2 chords each bar)
        - Texture (represented by note density)

        K-0 S-0 INS  i-2 i-26 i-128
        CD CR-0 CT-1 CD CR-2 CT-10  (chord of 1st bar)
        TF o-0 TF-8 o-6 TF-4 o-10 TF 10 (1 TF token for each o-n position token)
        CD CR-0 CT-1 CD CR-2 CT-10  (chord of 2nd bar)
        TF o-0 TF-8 o-6 TF-4 o-10 TF 10
        b-1 (end of 1st bar’s condition)
        … (conditions of the rest 7 bars)
        """
        data_dir = self.output_dir
        meta_path = jpath(data_dir, "metadata_resplit.json")
        meta = read_json(meta_path)
        chord_tk = ChordTokenizer()

        pbar = tqdm(meta)
        # for split in meta:
        #     split_entry = meta[split]

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]
            split = song_entry["split"]

            """
            Input sequence format:
            (info of entire sample)
                (key info)                          K-0
            (info of first bar) 
                (inst info)                         INS  INS-0  INS-3   (id small to large)
                    (info of first beat)
                        (chord)                     CD CR-0 CT-1 
                        (time function)             TF TF-8  TF-0  TF-3 (12 TF tokens) 
                    ...
                    (info of rest 3 beats)  
                (bar line)                          b-1 
            ...
            (conditions of the rest 7 bars)
            """
            input_tokens = []

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)

            b_1_indexes = [
                index for index, element in enumerate(remi_seq) if element == "b-1"
            ]
            num_bars = len(b_1_indexes)
            # to facilitate access first token of each bar, add a b-1 at the very beginning
            b_1_indexes.insert(0, -1)
            sample_dir = os.path.dirname(remi_fp)

            # print(num_bars)
            if num_bars == 0:
                raise Exception("Bar num = 0")
            # continue

            # Key
            key = song_entry["key"]  # major | minor

            # # Instrument

            # # print(sample_dir)
            # inst_fp = jpath(sample_dir, 'inst_per_bar.txt')
            # inst_per_bar = read_json(inst_fp)

            # Chord
            chords_per_bar = 2
            song_dir = os.path.dirname(remi_fp)
            chord_seg_fp = jpath(song_dir, "chord_from_recon.txt")
            with open(chord_seg_fp) as f:
                chords = f.read().strip().split(" ")

            # time_func_fp = jpath(sample_dir, 'time_func.txt')
            # time_func = read_json(time_func_fp)

            # Collate
            # Add Key
            if key == "major":
                key_token = "K-0"
            else:
                key_token = "K-1"
            input_tokens.append(key_token)

            # Add scale
            scale_token = "S-1"
            input_tokens.append(scale_token)

            # Iterate over all bars
            for bar_id in range(num_bars):
                bar_start_idx = b_1_indexes[bar_id] + 1
                bar_end_idx = b_1_indexes[bar_id + 1]
                bar_seq = remi_seq[bar_start_idx:bar_end_idx]

                """ Add Inst """
                # Obtain instruments from bar_seq
                insts_this_bar = set()
                for token in bar_seq:
                    if token.startswith("i-"):
                        insts_this_bar.add(token)
                insts_this_bar = list(insts_this_bar)
                insts_this_bar = sorted(
                    insts_this_bar, key=lambda x: int(x.split("-")[1])
                )  # sort by inst id

                # Add instrument info to input_tokens
                input_tokens.append("INS")
                input_tokens.extend(insts_this_bar)

                # Obtain chord and time func
                chord_idx = bar_id * chords_per_bar
                chord_this_bar = chords[chord_idx : chord_idx + chords_per_bar]

                if len(chord_this_bar) < 1:
                    chord_this_bar.append(
                        "N:N"
                    )  # Ensure each bar have at lease one chord symbol
                if len(chord_this_bar[0]) == 0:
                    chord_this_bar[0] = "N:N"

                # print(chord_this_bar, len(chord_this_bar))
                chord_tokenized = chord_tk.tokenize_chord_list(chord_this_bar)
                # time_func_this_bar = time_func[bar_id_str].split(' ')

                """ For each 2-beat, add chord """
                for beat_id in range(2):  # for each beat, id 0~3
                    # Chord tokens of the bar
                    input_tokens.append("CD")
                    chord_tokens_per_beat = 2
                    input_tokens.extend(
                        chord_tokenized[
                            beat_id
                            * chord_tokens_per_beat : beat_id
                            * chord_tokens_per_beat
                            + chord_tokens_per_beat
                        ]
                    )

                """ Time function of the bar """
                onset_density = get_time_function_from_remi_one_bar(bar_seq)     # Obtain onset count for each position
                txt_seq = get_time_function_from_remi_one_bar(bar_seq)
                input_tokens.extend(txt_seq)

                # Add bar line
                input_tokens.append("b-1")

            input_tokens_str = " ".join(input_tokens)
            tokenized_condition_fp = jpath(
                sample_dir, "tokenized_condition_key_inst_chd.txt"
            )
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            meta[song]["tokenized_condition"] = input_tokens_str

        save_json(meta, meta_path)

    def tokenize_for_source_separation(self):
        """
        Prepare the tokenized sequence for sybolic source separation training.
        Input: modified target sequence, retains only position, pitch, and bar line tokens
            o-12 p-172 o-24 i-64 o-36 p-172 b-1 o-0 p-68 ...
        Output: target sequence:
            o-12 i-128 p-172 d-15 o-24 i-64 p-68 d-9 o-36 i-128 p-172 d-15 b-1 o-0 i-64 p-68 d-9 ...
        """
        meta_path = self.meta_fp
        meta = read_json(meta_path)

        pbar = tqdm(meta)

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]

            # Read song remi seq
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)

            ''' Obtain input tokens from segment remi_seq '''
            input_tokens = remi_utils.obtain_input_tokens_from_remi_seg_for_sss(remi_seq)

            # Save results
            sample_dir = os.path.dirname(remi_fp)
            input_tokens_str = " ".join(input_tokens)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_sss_input.txt")
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            meta[song]["tokenized_condition_fp"] = tokenized_condition_fp

        save_json(meta, meta_path)

    def tokenize_for_sss_inst_pitch_pos(self):
        """
        Prepare the tokenized sequence for sybolic source separation training with decomposed pitch and rhythm

        Input: modified target sequence,
            - Separate instrument, pitch, and position, into different sub-sequences
            - Removed duration tokens
            INS i-5 i-8 PITCH p-172 p-32 o-12 p-172 TF o-24 o-36 b-1 ...
        Output: target sequence:
            o-12 i-128 p-172 d-15 o-24 i-64 p-68 d-9 o-36 i-128 p-172 d-15 b-1 o-0 i-64 p-68 d-9 ...
        """
        data_dir = self.output_dir
        meta_path = self.meta_fp
        meta = read_json(meta_path)
        chord_tk = ChordTokenizer()

        pbar = tqdm(meta)

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)

            """ Obtain Input tokens from segment remi_seq """
            input_tokens = self.obtain_input_tokens_from_remi_seq_sss_ipo(remi_seq)

            # Save results
            input_tokens_str = " ".join(input_tokens)
            sample_dir = os.path.dirname(remi_fp)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_sss_ipo_input.txt")
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            meta[song]["tokenized_condition"] = input_tokens_str

        save_json(meta, meta_path)

    def tokenize_for_sss_ipo_sort_pitch(self):
        """
        Prepare the tokenized sequence for sybolic source separation training with decomposed pitch and rhythm

        Input: modified target sequence,
            - Separate instrument, pitch, and position, into different sub-sequences
            - Removed duration tokens
            INS i-5 i-8 PITCH p-172 p-32 o-12 p-172 TF o-24 o-36 b-1 ...
        Output: target sequence:
            o-12 i-128 p-172 d-15 o-24 i-64 p-68 d-9 o-36 i-128 p-172 d-15 b-1 o-0 i-64 p-68 d-9 ...
        """
        data_dir = self.output_dir
        meta_path = self.meta_fp
        meta = read_json(meta_path)
        chord_tk = ChordTokenizer()

        pbar = tqdm(meta)

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)

            """ Obtain Input tokens from segment remi_seq """
            input_tokens = self.obtain_input_tokens_from_remi_seq_sss_ipo_sort_pitch(remi_seq)

            # Save results
            input_tokens_str = " ".join(input_tokens)
            sample_dir = os.path.dirname(remi_fp)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_sss_ipo_ps_input.txt")
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            meta[song]["tokenized_condition"] = input_tokens_str

        save_json(meta, meta_path)

    def tokenize_for_sss_ipo_with_onset_count(self):
        """
        Prepare the tokenized sequence for sybolic source separation training from subsequences of instrments, pitch, rhythm, and onset count

        Input: modified target sequence,
            - Separate instrument, pitch, position with onset count, into different sub-sequences
            - Removed duration tokens
            - Onset count are normalized to heavy beat (TF-2) and light beat (TF-1), within each bar
            INS i-5 i-8 PITCH p-172 p-32 o-12 p-172 TF o-24 TF-2 o-36 TF-1 b-1 ...
        Output: target sequence:
            o-12 i-128 p-172 d-15 o-24 i-64 p-68 d-9 o-36 i-128 p-172 d-15 b-1 o-0 i-64 p-68 d-9 ...
        """
        meta_path = self.meta_fp
        meta = read_json(meta_path)

        pbar = tqdm(meta)

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)

            """ Obtain Input tokens from segment remi_seq """
            input_tokens = self.obtain_input_tokens_from_remi_sss_ipo_tf(remi_seq)

            # Save results
            input_tokens_str = " ".join(input_tokens)
            sample_dir = os.path.dirname(remi_fp)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_sss_ipo_ps_input.txt")
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            meta[song]["tokenized_condition"] = input_tokens_str

        save_json(meta, meta_path)

    def tokenize_for_sss_ipo_tf_with_history(self):
        """
        Prepare the tokenized sequence for sybolic source separation training from subsequences of instruments, pitch, rhythm and onset count, and history

        Input: modified target sequence,
            - Separate instrument, pitch, position with onset count, into different sub-sequences
            - Removed duration tokens
            - Onset count are normalized to heavy beat (TF-2) and light beat (TF-1), within each bar
            - 1-bar history
            INS i-5 i-8 PITCH p-172 p-32 o-12 p-172 TF o-24 TF-2 o-36 TF-1 HIST o-0 i-2 p-36 d-32 ...
        Output: target sequence:
            o-12 i-128 p-172 d-15 o-24 i-64 p-68 d-9 o-36 i-128 p-172 d-15 b-1 o-0 i-64 p-68 d-9 ...
        """
        meta_path = self.meta_fp
        meta = read_json(meta_path)

        pbar = tqdm(meta)

        for song in pbar:
            pbar.set_description(song)
            song_entry = meta[song]

            # Remi
            remi_fp = song_entry["remi"]
            remi_seq = read_remi(remi_fp)

            """ Obtain Input tokens from segment remi_seq """
            # Need ensure remi_seq contain strictly 2 bar of info
            input_tokens = remi_utils.obtain_input_tokens_from_remi_sss_ipo_tf_hist(remi_seq)
            tgt_seq = remi_utils.obtain_target_tokens_from_remi_sss_ipo_tf_hist(remi_seq)

            # Save results
            input_tokens_str = " ".join(input_tokens)
            sample_dir = os.path.dirname(remi_fp)
            tokenized_condition_fp = jpath(sample_dir, "tokenized_sss_ipo_tf_hist_input.txt")
            with open(tokenized_condition_fp, "w") as f:
                f.write(input_tokens_str + "\n")
            tgt_seq_str = ' '.join(tgt_seq)
            tgt_seq_fp = jpath(sample_dir, 'tokenized_sss_ipo_tf_hist_tgt.txt')
            with open(tgt_seq_fp, 'w') as f:
                f.write(tgt_seq_str + '\n')
            meta[song]["tokenized_condition_fp"] = tokenized_condition_fp
            meta[song]['tgt_seq_fp'] = tgt_seq_fp


        save_json(meta, meta_path)

    def obtain_input_tokens_from_remi_seq_sss_ipo(self, remi_seq):
        input_tokens = []
        from utils_midi.utils_midi import RemiUtil

        b_1_indices = RemiUtil.get_bar_idx_from_remi(remi_seq)
        num_bars = len(b_1_indices)

        if num_bars == 0:
            raise Exception("Bar num = 0")

        # Iterate over all bars
        for bar_id in b_1_indices:
            bar_start_idx, bar_end_idx = b_1_indices[bar_id]
            bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

            inst_tokens = set()
            pitch_tokens = []
            pos_tokens = []

            """ Only retain position, pitch, and bar line """
            for tok in bar_remi_seq:
                if tok.startswith("i-"):
                    inst_tokens.add(tok)
                elif tok.startswith("p-"):
                    pitch_tokens.append(tok)
                elif tok.startswith("o-"):
                    pos_tokens.append(tok)

            # Convert inst token to list
            inst_tokens = list(inst_tokens)
            inst_tokens = sorted(
                inst_tokens, key=lambda x: int(x.split("-")[1])
            )  # sort by inst id

            input_tokens.append("INS")
            input_tokens.extend(inst_tokens)
            input_tokens.append("PITCH")
            input_tokens.extend(pitch_tokens)

            
            input_tokens.append("TF")
            input_tokens.extend(pos_tokens)

            # Add a bar line token in the end
            input_tokens.append("b-1")

        return input_tokens

    def obtain_input_tokens_from_remi_sss_ipo_tf(self, remi_seq):
        input_tokens = []
        from utils_midi.utils_midi import RemiUtil

        b_1_indices = RemiUtil.get_bar_idx_from_remi(remi_seq)
        num_bars = len(b_1_indices)

        if num_bars == 0:
            raise Exception("Bar num = 0")

        # Iterate over all bars
        for bar_id in b_1_indices:
            bar_start_idx, bar_end_idx = b_1_indices[bar_id]
            bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

            inst_tokens = set()
            pitch_tokens = []

            """ Only retain position, pitch, and bar line """
            for tok in bar_remi_seq:
                if tok.startswith("i-"):
                    inst_tokens.add(tok)
                elif tok.startswith("p-"):
                    pitch_tokens.append(tok)

            # Convert inst token to list
            inst_tokens = list(inst_tokens)
            inst_tokens = sorted(
                inst_tokens, key=lambda x: int(x.split("-")[1])
            )  # sort by inst id

            input_tokens.append("INS")
            input_tokens.extend(inst_tokens)
            input_tokens.append("PITCH")
            input_tokens.extend(pitch_tokens)

            onset_density = get_onset_density_of_a_bar_from_remi(bar_remi_seq)
            txt_tokens = tokenize_onset_density_one_bar(onset_density, quantize=True)
            input_tokens.append("TF")
            input_tokens.extend(txt_tokens)

            # Add a bar line token in the end
            input_tokens.append("b-1")

        return input_tokens
    
    def obtain_input_tokens_from_remi_seq_sss_ipo_sort_pitch(self, remi_seq):
        '''
        In previous segmented dataset for sss ipo, the order of pitch token may give hint to the instrumentation.
        Validate if this is true or not, by putting all pitch in the same position from a strict high-to-low order.
        '''
        input_tokens = []
        from utils_midi.utils_midi import RemiUtil

        b_1_indices = RemiUtil.get_bar_idx_from_remi(remi_seq)
        num_bars = len(b_1_indices)

        if num_bars == 0:
            raise Exception("Bar num = 0")

        # Iterate over all bars
        for bar_id in b_1_indices:
            bar_start_idx, bar_end_idx = b_1_indices[bar_id]
            bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

            inst_tokens = set()
            pitch_tokens = []
            pos_tokens = []

            cur_pos = None
            pitch_of_the_pos = None
            """ Only retain position, pitch, and bar line """
            for tok in bar_remi_seq:
                if tok.startswith("i-"):
                    inst_tokens.add(tok)
                elif tok.startswith("p-"):
                    # pitch_tokens.append(tok)
                    pitch_of_the_pos.append(tok)
                elif tok.startswith("o-"):
                    # Add the pich of previous position to pitch seq
                    if pitch_of_the_pos != None:
                        pitch_of_the_pos = sorted(
                                pitch_of_the_pos, key=lambda x: int(x.split("-")[1]), reverse=True
                            )  # Sort pitch from high to low
                        pitch_tokens.extend(pitch_of_the_pos)

                    pos_tokens.append(tok)
                    cur_pos = int(tok.split('-')[-1])
                    pitch_of_the_pos = []
            # Add the pitch of the last position to pitch seq
            if pitch_of_the_pos != None:
                pitch_of_the_pos = sorted(
                        pitch_of_the_pos, key=lambda x: int(x.split("-")[1]), reverse=True
                    )  # Sort pitch from high to low
                pitch_tokens.extend(pitch_of_the_pos)

            # Convert inst token to list
            inst_tokens = list(inst_tokens)
            inst_tokens = sorted(
                inst_tokens, key=lambda x: int(x.split("-")[1])
            )  # sort by inst id

            input_tokens.append("INS")
            input_tokens.extend(inst_tokens)
            input_tokens.append("PITCH")
            input_tokens.extend(pitch_tokens)
            input_tokens.append("TF")
            input_tokens.extend(pos_tokens)

            # Add a bar line token in the end
            input_tokens.append("b-1")

        return input_tokens

    def observe_sequence_length(self, quantile=1.0):
        """
        Check the length of sequence in both src and tgt data.
        """
        data_dir = jpath(self.output_dir, "collated")
        meta = read_json(self.meta_fp)
        out_dir = jpath(data_dir, "statistics")
        create_dir_if_not_exist(out_dir)
        sample_cnt = {}

        for split in ["train", "valid", "test"]:
            print(split)

            tot_token_cnt = []
            src_fn = split + "_input.txt"  # 重新统计长度
            src_fp = jpath(data_dir, src_fn)
            tgt_fn = "{}.txt".format(split)
            tgt_fp = jpath(data_dir, tgt_fn)
            with open(src_fp) as f:
                src_data = f.readlines()
            with open(tgt_fp) as f:
                tgt_data = f.readlines()
            tot_seq_len_list = []
            src_seq_len_list = []
            tgt_seq_len_list = []
            for src_sent, tgt_sent in zip(src_data, tgt_data):
                src_seq = src_sent.strip().split(" ")
                tgt_seq = tgt_sent.strip().split(" ")
                tot_len = len(src_seq) + len(tgt_seq)
                tot_seq_len_list.append(tot_len)
                src_seq_len_list.append(len(src_seq))
                tgt_seq_len_list.append(len(tgt_seq))

            # sample_cnt['{}'.format(split)] = num_samples
            print("Src sequence length:")
            t = np.quantile(src_seq_len_list, quantile)
            print("{}: {}".format(split, t))

            print("Tgt sequence length:")
            t = np.quantile(tgt_seq_len_list, quantile)
            print("{}: {}".format(split, t))

            print("tot sequence length:")
            t = np.quantile(tot_seq_len_list, quantile)
            print("{}: {}".format(split, t))
            print()

        out_fp = jpath(out_dir, "sample_cnt.json")
        save_json(sample_cnt, out_fp)

    def observe_num_inst(self, quantile=1.0):
        """
        Check the length of sequence in both src and tgt data.
        """
        data_dir = jpath(self.output_dir, "collated")
        meta = read_json(self.meta_fp)
        out_dir = jpath(data_dir, "statistics")
        create_dir_if_not_exist(out_dir)
        sample_cnt = {}

        for split in ["train", "valid", "test"]:
            print(split)

            src_fn = split + "_input.txt"  # 重新统计长度
            src_fp = jpath(data_dir, src_fn)
            tgt_fn = "{}.txt".format(split)
            tgt_fp = jpath(data_dir, tgt_fn)
            with open(src_fp) as f:
                src_data = f.readlines()
            with open(tgt_fp) as f:
                tgt_data = f.readlines()
            inst_nums = []
            for src_sent in src_data:
                insts = set([i for i in src_sent.strip().split(" ") if i.startswith('i-')])
                inst_num = len(insts)
                inst_nums.append(inst_num)

            # sample_cnt['{}'.format(split)] = num_samples
            print("Src sequence length:")
            t = np.quantile(inst_nums, quantile)
            print("{}: {}".format(split, t))

            print()

        out_fp = jpath(out_dir, "sample_cnt.json")
        save_json(sample_cnt, out_fp)
    
    def collate_into_single_files(self, len_limit=99999):
        """
        Collate all input and output from all samples into a single file.
        Note: won't include samples that exceed the upper length limit
        """
        data_dir = self.output_dir
        meta_path = jpath(data_dir, "metadata_resplit.json")
        meta = read_json(meta_path)
        chord_tk = ChordTokenizer()
        out_dir = jpath(data_dir, "collated")
        create_dir_if_not_exist(out_dir)

        for split_name in ["test", "valid", "train"]:
            print("Collating {} set".format(split_name))

            src_data = []
            tgt_data = []
            sample_id = 0
            track_names = []  # map from id to track names

            pbar = tqdm(meta)
            for song in pbar:
                pbar.set_description(song)
                song_entry = meta[song]
                split = song_entry["split"]
                if split == "validation":
                    split = "valid"

                if split == split_name:
                    remi_fp = song_entry["remi"]
                    # tgt_seq_fp = song_entry['tgt_seq_fp']
                    condition_song_fp = song_entry["tokenized_condition_fp"]
                    remi_song_seq = read_remi(remi_fp, split=True)
                    inp_song_seq = read_remi(condition_song_fp, split=True)

                    # Length check
                    if len(remi_song_seq) + len(inp_song_seq) > len_limit:
                        continue

                    # Collate target sequence
                    tgt_song_str = ' '.join(remi_song_seq)
                    tgt_data.append(tgt_song_str + "\n")

                    # Collate input data
                    inp_song_str = ' '.join(inp_song_seq)
                    src_data.append(inp_song_str + "\n")

                    # Record the map between id and segment name
                    track_names.append(song)

                    sample_id += 1

            split_out_fp = jpath(out_dir, "{}.txt".format(split_name))
            with open(split_out_fp, "w") as f:
                f.writelines(tgt_data)

            split_out_fp = jpath(out_dir, "{}_input.txt".format(split_name))
            with open(split_out_fp, "w") as f:
                f.writelines(src_data)

            id_to_track_info_dir = jpath(out_dir, "id_to_track_name")
            create_dir_if_not_exist(id_to_track_info_dir)
            id_to_track_name_fp = jpath(id_to_track_info_dir, f"{split_name}.json")
            save_json(track_names, id_to_track_name_fp)

    def de_tokenize(self):
        """
        De-tokenize remi segment to MIDI to facilitate alignment checking.
        Move both remi and MIDI to same folder
        Modify the entry of remi path
        Add entry of MIDI path to metadata
        """
        data_dir = jpath(self.output_dir, "data")

        meta_path = self.meta_fp
        meta = read_json(meta_path)
        tk = RemiTokenizer()
        pbar = tqdm(meta)
        for segment_name in pbar:
            pbar.set_description(segment_name)
            song_entry = meta[segment_name]
            split = song_entry["split"]
            remi_fp = song_entry["remi"]
            song_name = segment_name.split(".")[0]
            song_dir = os.path.dirname(remi_fp)
            if not os.path.exists(song_dir):
                raise Exception("Dir of segment sample not exist: {}".format(song_dir))

            # Detokenize to MIDI
            midi_fp = jpath(song_dir, "midi.mid")
            with open(remi_fp) as f:
                remi_str = f.readline().strip()
            remi_seq = remi_str.split(" ")
            tk.remi_to_midi(remi_seq, midi_fp)

            # # Update metadata
            # meta[song]['remi_fp'] = remi_fp_new
            # meta[song]['midi_fp'] = midi_fp

        # meta_path_new = jpath(data_dir, 'metadata_new.json')
        # save_json(meta, meta_path_new)

    def obtain_chord_from_segment(self):
        """
        To debug the chord type recognition, directly obtain the chord from the segment.
        Chords were originally detected from song-level data,
        Put accuracy aside, at least this is a different scheme than test time.
        Because in evaluation, chord are recognized from reconstructed midi from output remi seq.
        """
        data_dir = jpath(self.output_dir, "data")

        from utils_chord.chord_map import recognize_chord_from_midi, quantize_chord
        from utils_midi.utils_midi import MidiUtil

        meta_path = self.meta_fp
        meta = read_json(meta_path)
        tk = RemiTokenizer()

        songs = list(meta.keys())
        songs.sort()
        pbar = tqdm(meta)
        for segment_name in pbar:
            pbar.set_description(segment_name)

            song_entry = meta[segment_name]
            split = song_entry["split"]
            remi_fp = song_entry["remi"]
            song_name = segment_name.split(".")[0]
            song_dir = os.path.dirname(remi_fp)

            # Recognize chord from detokenized midi
            midi_fp = jpath(song_dir, "midi.mid")
            if not os.path.exists(midi_fp):
                raise Exception("Midi file path {} not exist.".format(midi_fp))

            chords = recognize_chord_from_midi(midi_fp, out_fp=None)
            dur = MidiUtil.get_duration(midi_fp)
            chord_quantized = quantize_chord(
                chords, dur=dur, num_bars=2, num_chord_per_bar=2
            ).tolist()

            out_fp = jpath(song_dir, "chord_from_recon.txt")
            with open(out_fp, "w") as f:
                f.write(" ".join(chord_quantized))

    def compare_chord_difference(self):
        """
        Compare the chord annotation obtained from segment and song
        """
        data_dir = jpath(self.output_dir, "data")

        from utils_chord.chord_map import recognize_chord_from_midi, quantize_chord
        from utils_midi.utils_midi import MidiUtil
        from evaluate import Metric

        meta_path = self.meta_fp
        meta = read_json(meta_path)
        tk_remi = RemiTokenizer()
        tk_chord = ChordTokenizer()

        metric = Metric()

        songs = list(meta.keys())
        songs.sort()
        pbar = tqdm(meta)
        for segment_name in pbar:
            pbar.set_description(segment_name)

            song_entry = meta[segment_name]
            split = song_entry["split"]
            remi_fp = song_entry["remi"]
            song_name = segment_name.split(".")[0]
            song_dir = os.path.dirname(remi_fp)

            # Obtain tokenized chord from segment
            chord_seg_fp = jpath(song_dir, "chord_from_recon.txt")
            with open(chord_seg_fp) as f:
                chord_seg = f.read().strip().split(" ")
            chord_seg = tk_chord.tokenize_chord_list(chord_seg)
            chord_seg_root = [i for i in chord_seg if i.startswith("CR")]
            chord_seg_type = [i for i in chord_seg if i.startswith("CT")]

            # Obtain tokenized chord from song-level recognition
            conditions_fp = jpath(song_dir, "tokenized_condition.txt")
            with open(conditions_fp) as f:
                conditions = f.read().strip().split(" ")
            chord_song = [
                i for i in conditions if i.startswith("CR") or i.startswith("CT")
            ]
            chord_song_root = [i for i in chord_song if i.startswith("CR")]
            chord_song_type = [i for i in chord_song if i.startswith("CT")]

            chord_root_acc = Metric.calculate_output_accuracy(
                chord_seg_root, chord_song_root
            )
            chord_type_acc = Metric.calculate_output_accuracy(
                chord_seg_type, chord_song_type
            )
            metric.update("chord_root_acc", chord_root_acc)
            metric.update("chord_type_acc", chord_type_acc)
            a = 1

        res = metric.average()
        print(res)


if __name__ == "__main__":
    _main()

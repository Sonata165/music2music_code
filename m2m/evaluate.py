'''
Calculate the objective metrics from HF models
'''

import os
import sys

sys.path.append('..')

import numpy as np
from torch import utils
from utils_common.utils import read_yaml, jpath, get_latest_checkpoint, save_json
from utils_midi import remi_utils
from utils_midi.remi_utils import from_target_bar_obtain_features, from_remi_get_pitch_seq_per_track, from_remi_get_pitch_seq_global
from tqdm import tqdm
import mlconfig
import math
from sklearn.metrics.pairwise import cosine_similarity
from evaluations.piano_evaluator import bar_level_note_f1_from_proll
import torch


def main():
    if len(sys.argv) == 2:
        config_fp = sys.argv[1]
        # config = read_yaml(config_fp)
        config = mlconfig.load(config_fp)
    else:
        config_fn = 'ar_hist.yaml'
        config_fp = '../src_hf/hparams/arrangement/{}'.format(config_fn)
        # config = read_yaml(config_fp)
        config = mlconfig.load(config_fp)

    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    out_fn = config['infer_out_fn']
    out_fp = jpath(out_dir, 'lightning_logs', latest_version_dir, out_fn)
    with open(out_fp) as f:
        out = f.readlines()
    out = [l.strip() for l in out]

    # Prepare test set
    data_root = config['data_root']
    data_fn = config['infer_inp_fn']
    data_fp = jpath(data_root, data_fn)
    with open(data_fp) as f:
        ref = f.readlines()
    ref = [l.strip() for l in ref]

    # Get the part of data for comparison
    new_out = []
    new_ref = []
    conditions = []
    out_feats = []
    for out_l, ref_l in zip(out, ref):
        out_l_seq = out_l.split(' ')
        ref_l_seq = ref_l.split(' ')

        if config['remove_drum'] is True:
            ref_l_seq = remi_utils.from_remi_bar_remove_drum(ref_l_seq)

        # Obtain original conditions
        sep_pos = out_l_seq.index('<sep>')
        condition_seq = out_l_seq[:sep_pos]
        # condition = get_condition_from_condition_seq(condition_seq)
        condition = condition_seq
        conditions.append(condition)

        # Obtain output seq
        out_l_seq = out_l_seq[sep_pos+1:]
        if out_l_seq[-1] == '</s>':
            out_l_seq = out_l_seq[:-1]

        # Calculate output features
        output_feat = from_target_bar_obtain_features(out_l_seq)
        out_feats.append(output_feat)
        
        # Obtain reference seq
        first_bar_line_pos = ref_l_seq.index('b-1')
        ref_l_seq = ref_l_seq[first_bar_line_pos+1:]

        new_out.append(out_l_seq)
        new_ref.append(ref_l_seq)
    outs = new_out
    refs = new_ref

    # Compute metrics
    res = []
    metric = Metric()
    for condition, ref, out, out_feat in tqdm(zip(conditions, refs, outs, out_feats), total=len(conditions)):
        t = {
            'condition': condition,
            'ref': ref,
            'out': out,
            'out_feat': out_feat,
        }

        # Instrument IOU
        #  # Old ver: not suitable for rand inst infer
        #  # Get instrument sequences from the output and target
        # inst_iou = metric.calculate_inst_iou(out, ref)
        # metric.update('bar_inst_iou', inst_iou, extend=True)
        # t['bar_inst_iou'] = inst_iou

        # [06-08 ver] New ver: suitable for rand inst infer
        inst_iou = metric.calculate_inst_iou_from_condition(out, condition)
        metric.update('bar_inst_iou', inst_iou)
        t['bar_inst_iou'] = inst_iou

        # # Content SOR (computed on the content) (deprecated)
        # # TODO: tgt content is not correctly computed for sort inst setting
        # content_out = out_feat['pitch_seq']
        # content_tgt = condition['pitch_seq']
        # content_sor = metric.calculate_sor(content_out, content_tgt)
        # metric.update('content_sor', content_sor)
        # t['content_sor'] = content_sor

        ''' Content SOR '''
        # Note: same note are removed from pitch sequence
        out_p_seq = remi_utils.from_remi_get_pitch_seq_flattened(out)
        tgt_p_seq = remi_utils.from_remi_get_pitch_seq_flattened(ref)
        pitch_sor = metric.calculate_sor(out_p_seq, tgt_p_seq)
        # pitch_sor = metric.calculate_pitch_sor(out_seq=out, tgt_seq=ref)
        metric.update('bar_pitch_sor', pitch_sor)
        t['bar_pitch_sor'] = pitch_sor

        pitch_wer = metric.calculate_wer(out_p_seq, tgt_p_seq)
        metric.update('bar_pitch_wer', pitch_wer)
        t['bar_pitch_wer'] = pitch_wer

        # Evaluate voice of instruments
        tgt_inst_seq = remi_utils.from_remi_get_inst_and_voice(ref)
        out_inst_seq = remi_utils.from_remi_get_inst_and_voice(out)
        voice_wer = metric.calculate_wer(out_inst_seq, tgt_inst_seq)
        t['bar_voice_wer'] = voice_wer
        metric.update('bar_voice_wer', voice_wer)

        ''' Position prediction (groove) '''
        pos_wer, pos_sor = metric.calculate_groove_wer_sor(out, ref)
        t['bar_groove_wer'] = pos_wer
        metric.update('bar_groove_wer', pos_wer)
        t['bar_groove_sor'] = pos_sor
        metric.update('bar_groove_sor', pos_sor)

        # Evaluate melody keeping
        tgt_melody = remi_utils.from_remi_get_melody_pitch_seq_highest_pitch(ref)
        out_melody = remi_utils.from_remi_get_melody_pitch_seq_highest_pitch(out)
        melody_wer = metric.calculate_wer(out_melody, tgt_melody)
        t['bar_melody_wer'] = melody_wer
        metric.update('bar_melody_wer', melody_wer)

        melody_sor = metric.calculate_melody_sor(out, ref)
        t['bar_melody_sor'] = melody_sor
        metric.update('bar_melody_sor', melody_sor)

        ''' Evaluate track-wise pitch range '''
        tgt_range_per_track = remi_utils.from_remi_get_range_of_track_dict(ref, return_int=True)
        out_range_per_track = remi_utils.from_remi_get_range_of_track_dict(out, return_int=True)
        # Calculate the absolute difference of lower bound, between the output and target for each shared instrument
        lower_bound_diff = {k: abs(out_range_per_track[k][0] - tgt_range_per_track[k][0]) for k in out_range_per_track if k in tgt_range_per_track}
        # Calculate the absolute difference of upper bound, between the output and target for each shared instrument
        upper_bound_diff = {k: abs(out_range_per_track[k][1] - tgt_range_per_track[k][1]) for k in out_range_per_track if k in tgt_range_per_track}
        # Average the differences
        # if len(tgt_range_per_track) == 0:
        #     range_diff = 0
        # elif len(out_range_per_track) == 0:
        #     range_diff = 1
        if len(lower_bound_diff) != 0:
            range_diff = (sum(lower_bound_diff.values()) + sum(upper_bound_diff.values())) / (len(lower_bound_diff) + len(upper_bound_diff))
            range_diff /= 100 # will be x100 in the end
            t['track_range_diff'] = range_diff
            metric.update('track_range_diff', range_diff)


        # Track-wise pitch SOR (for overlapped tracks)
        track_p_sor = metric.calculate_avg_track_sor(out_seq=out, tgt_seq=ref)
        t['track_sor'] = track_p_sor
        metric.update('track_sor', track_p_sor)

        # Evaluate texture of each track
        track_texture_acc = metric.calculate_avg_track_texture_acc(out, ref)
        t['track_texture_acc'] = track_texture_acc
        metric.update('track_texture_acc', track_texture_acc)

        ''' Evaluate groove of each instrument '''
        tgt_groove = remi_utils.from_remi_get_pos_per_track_dict(ref)
        out_groove = remi_utils.from_remi_get_pos_per_track_dict(out, remi_reordered=config['reorder_tgt'])
        # Compute WER for shared instruments
        groove_wers = {}
        for inst in tgt_groove:
            if inst in out_groove:
                groove_wer = metric.calculate_wer(out_groove[inst], tgt_groove[inst])
                groove_wers[inst] = groove_wer
        # Average the WERs
        if len(groove_wers) == 0:
            avg_groove_wer = 0
        else:
            avg_groove_wer = sum(groove_wers.values()) / len(groove_wers)
        t['track_groove_wer'] = avg_groove_wer
        metric.update('track_groove_wer', avg_groove_wer)

        ''' Evaluate bass prediction '''
        tgt_bass_pitch_seq = remi_utils.from_remi_get_bass_pitch_seq(ref)
        out_bass_pitch_seq = remi_utils.from_remi_get_bass_pitch_seq(out)
        bass_wer = metric.calculate_wer(out_bass_pitch_seq, tgt_bass_pitch_seq)
        t['bass_wer'] = bass_wer
        metric.update('bass_wer', bass_wer)

        ''' Evaluate drum prediction '''
        tgt_drum_pitch_seq = remi_utils.from_remi_get_drum_content_seq(ref)
        out_drum_pitch_seq = remi_utils.from_remi_get_drum_content_seq(out)
        drum_wer = metric.calculate_wer(out_drum_pitch_seq, tgt_drum_pitch_seq)
        t['drum_wer'] = drum_wer
        metric.update('drum_wer', drum_wer)

        ''' Evaluate duration prediction '''
        # NOTE: duration token share the same unit as position token, i.e., 1/12 beat
        tgt_duration = remi_utils.from_remi_get_avg_duration_per_track(ref)
        out_duration = remi_utils.from_remi_get_avg_duration_per_track(out)
        # Calculate the absolute difference between the output and target for each shared instrument
        duration_diff = {k: abs(out_duration[k] - tgt_duration[k]) for k in out_duration if k in tgt_duration}
        # if len(tgt_duration) == 0:
        #     if len(out_duration) == 0:
        #         avg_duration_diff = 0
        #     else:
        #         # Average the values of the output duration as the difference
        #         avg_duration_diff = sum(out_duration.values()) / len(out_duration)
        # else:
        #     if len(out_duration) == 0:
        #         # Average the values of the target duration as the difference
        #         avg_duration_diff = sum(tgt_duration.values()) / len(tgt_duration)
        #     else:
        if len(duration_diff) != 0:
            avg_duration_diff = sum(duration_diff.values()) / len(duration_diff)
        
            # Change the unit to beat
            avg_duration_diff = avg_duration_diff / 12 / 100 # will be x100 in the end
            t['note_duration_diff'] = avg_duration_diff 
            metric.update('note_duration_diff', avg_duration_diff)

        res.append(t)

    res = metric.average()
    for k in res:
        res[k] = round(res[k], 2)
    print(res)

    # Save the results
    out_fn = config['eval_out_fn'] if 'infer_exp_name' in config else 'metrics_recon.txt'
    res_fp = jpath(out_dir, 'lightning_logs', latest_version_dir, out_fn)
    save_json(res, res_fp)

def get_condition_from_condition_seq(condition_seq):
    PITCH_pos = condition_seq.index('PITCH')
    INS_pos = condition_seq.index('INS')

    if 'HIST' in condition_seq:
        HIST_pos = condition_seq.index('HIST')
    else:
        HIST_pos = len(condition_seq)

    pitch_seq = condition_seq[PITCH_pos:INS_pos][1:]
    inst_seq = condition_seq[INS_pos:HIST_pos][1:]
    hist_seq = condition_seq[HIST_pos:][1:]

    ret = {
        'pitch_seq': pitch_seq,
        'inst_seq': inst_seq,
        'hist_seq': hist_seq
    }
    return ret

class Metric:
    def __init__(self):
        self.clear()

    def clear(self):
        self.metrics = {}

    def update(self, metric_name, metric_value, extend=False):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        if not extend:
            self.metrics[metric_name].append(metric_value)
        else:
            self.metrics[metric_name].extend(metric_value)

    def average(self):
        for metric_name in self.metrics:
            t = self.metrics[metric_name]
            self.metrics[metric_name] = sum(t) / len(t)
        ret = self.metrics

        for k in ret:
            ret[k] = round(ret[k], 5)

        self.clear()
        return ret

    @staticmethod
    def calculate_IoU_for_batched_lists(out, tgt):
        # 初始化IoU列表
        iou_scores = []

        # 确保输入的两个列表长度相同
        if len(out) != len(tgt):
            # raise Exception("Error: The length of 'out' and 'tgt' must be the same.")
            min_len = min(len(out), len(tgt))
            out = out[:min_len]
            tgt = tgt[:min_len]

        # 遍历out和tgt中的子列表
        for out_sublist, tgt_sublist in zip(out, tgt):
            # 计算交集
            intersection = set(out_sublist) & set(tgt_sublist)
            # 计算并集
            union = set(out_sublist) | set(tgt_sublist)
            # 计算IoU并添加到IoU列表
            if union:  # 防止除以0
                iou_scores.append(len(intersection) / len(union))
            else:
                iou_scores.append(1)  # 如果两个列表都是空的，则IoU为0

        return iou_scores

    @staticmethod
    def calculate_IoU_for_batched_dicts(out, tgt):
        """
        Calculate the Intersection over Union score for lists of dicts.
        IoU is calculated for each pair of k in out and tgt, and then averaged within each dict
        Return a list of scores containing scores for each dict.
        """
        # 初始化IoU列表
        iou_scores = []

        # 确保输入的两个列表长度相同
        if len(out) != len(tgt):
            # raise Exception("Error: The length of 'out' and 'tgt' must be the same.")
            min_len = min(len(out), len(tgt))
            out = out[:min_len]
            tgt = tgt[:min_len]

        # 遍历out和tgt中的dict
        for out_subdict, tgt_subdict in zip(out, tgt):
            # 初始化交集和并集的总计
            total_intersection = 0
            total_union = 0

            # 获取两个字典键的并集
            all_keys = set(out_subdict.keys()) | set(tgt_subdict.keys())

            for key in all_keys:
                out_count = out_subdict.get(key, 0)  # 如果键不存在，则计数为0
                tgt_count = tgt_subdict.get(key, 0)  # 如果键不存在，则计数为0

                # 交集为两个计数中的较小者
                intersection = min(out_count, tgt_count)
                # 并集为两个计数中的较大者
                union = max(out_count, tgt_count)

                # 更新总计
                total_intersection += intersection
                total_union += union

            # 计算平均IoU得分. 如果两个dict都为空，算作满分
            average_iou = total_intersection / total_union if total_union > 0 else 1
            iou_scores.append(average_iou)

        return iou_scores

    @staticmethod
    def calculate_output_accuracy(out, tgt):
        """
        计算两个列表中字符串元素的输出准确度。
        准确度是指两个列表在相同位置的字符串完全相同的比例。

        参数:
        - out: 输出列表，包含字符串元素。
        - tgt: 目标列表，包含字符串元素。

        返回:
        - 准确度：一个介于0和1之间的浮点数，表示匹配的比例。
        """
        # 确保输入列表长度相同
        if len(out) != len(tgt):
            min_len = min(len(out), len(tgt))
            out = out[:min_len]
            tgt = tgt[:min_len]
            # raise ValueError("The length of 'out' and 'tgt' must be the same.")

        # 计算匹配的元素数量
        matches = sum(1 for o, t in zip(out, tgt) if o == t)

        # 计算准确度
        accuracy = matches / len(out) if out else 0  # 如果列表为空，则准确度为0

        return accuracy
    
    @staticmethod
    def calculate_sor_batch(out_seqs, tgt_seqs):
        """
        Calculate sequence overlap ratio

        res = (number of tokens in output that overlap with tgt, under the best match) / min(|out|, |tgt|)
        """
        ret = []
        for out_seq, tgt_seq in zip(out_seqs, tgt_seqs):
            sor = Metric.calculate_sor(out_seq, tgt_seq)
            ret.append(sor)
        return ret

    @staticmethod
    def calculate_sor(out_seq, tgt_seq):
        """
        Calculate sequence overlap ratio

        res = (number of tokens in output that overlap with tgt) / min(|out|, |tgt|)
        """
        # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
        # The extra row and column are for the case when one of the strings is empty
        d = np.zeros((len(tgt_seq) + 1, len(out_seq) + 1))

        # The number of operations for an empty hypothesis to become the reference
        # is just the number of words in the reference (i.e., deleting all words)
        for i in range(len(tgt_seq) + 1):
            d[i, 0] = 0 # In SOR, deletion not count as operation
        # The number of operations for an empty reference to become the hypothesis
        # is just the number of words in the hypothesis (i.e., inserting all words)
        for j in range(len(out_seq) + 1):
            d[0, j] = j
        # Iterate over the words in the reference and hypothesis
        for i in range(1, len(tgt_seq) + 1):
            for j in range(1, len(out_seq) + 1):
                # If the current words are the same, no operation is needed
                # So we just take the previous minimum number of operations
                if tgt_seq[i - 1] == out_seq[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    # If the words are different, we consider three operations:
                    # substitution, insertion, and deletion
                    # And we take the minimum of these three possibilities
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j]    # Deletion不计入operation
                    d[i, j] = min(substitution, insertion, deletion)
        
        # The minimum number of operations to transform the hypothesis into the reference
        # is in the bottom-right cell of the matrix
        # We divide this by the number of words in the reference to get the WER
        # wer = d[len(tgt_seq), len(out_seq)] / len(tgt_seq)
        mismatch_token_num = d[len(tgt_seq), len(out_seq)]
        match_token_num = len(out_seq) - mismatch_token_num
        denom = min(len(out_seq), len(tgt_seq))

        if len(tgt_seq) == 0: # denom is 0, empty target
            if len(out_seq) != 0: # all insertion
                return 0
            else: # empty output
                return 1
        elif len(out_seq) == 0: # denom is 0, non-empty target, empty output, all deletion
            return 1 # deletion does not count
        else: # denom is not 0
            sor = match_token_num / denom

            return sor
    
    @staticmethod
    def calculate_wer_batch(out_seqs, tgt_seqs):
        """
        Calculate sequence overlap ratio

        res = (number of tokens in output that overlap with tgt) / min(|out|, |tgt|)
        """
        ret = []
        for out_seq, tgt_seq in zip(out_seqs, tgt_seqs):
            sor = Metric.calculate_wer(out_seq, tgt_seq)
            ret.append(sor)
        return ret

    @staticmethod
    def calculate_wer(out_seq, tgt_seq):
        """
        Calculate word error rate

        res = (number of tokens in output that overlap with tgt) / min(|out|, |tgt|)
        """
        # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
        # The extra row and column are for the case when one of the strings is empty
        d = np.zeros((len(tgt_seq) + 1, len(out_seq) + 1))
        # The number of operations for an empty hypothesis to become the reference
        # is just the number of words in the reference (i.e., deleting all words)
        for i in range(len(tgt_seq) + 1):
            d[i, 0] = i
        # The number of operations for an empty reference to become the hypothesis
        # is just the number of words in the hypothesis (i.e., inserting all words)
        for j in range(len(out_seq) + 1):
            d[0, j] = j
        # Iterate over the words in the reference and hypothesis
        for i in range(1, len(tgt_seq) + 1):
            for j in range(1, len(out_seq) + 1):
                # If the current words are the same, no operation is needed
                # So we just take the previous minimum number of operations
                if tgt_seq[i - 1] == out_seq[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    # If the words are different, we consider three operations:
                    # substitution, insertion, and deletion
                    # And we take the minimum of these three possibilities
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)
        
        # The minimum number of operations to transform the hypothesis into the reference
        # is in the bottom-right cell of the matrix
        # We divide this by the number of words in the reference to get the WER
        if len(tgt_seq) == 0: # if target empty
            if len(out_seq) != 0: # out not empty
                wer = 1 # all insertion error
            else: # out also empty
                wer = 0 # no error
        else: # target not empty
            wer = d[len(tgt_seq), len(out_seq)] / len(tgt_seq)
        
        return wer

    def calculate_melody_sor(self, out_seq, tgt_seq):
        tgt_melody = remi_utils.from_remi_get_melody_seq(tgt_seq)
        out_melody = remi_utils.from_remi_get_melody_seq(out_seq)
        melody_sor = self.calculate_sor(out_melody, tgt_melody)
        return melody_sor

    def calculate_dur_dif_per_track(self, out_seq, tgt_seq):
        tgt_duration = remi_utils.from_remi_get_avg_duration_per_track(tgt_seq)
        out_duration = remi_utils.from_remi_get_avg_duration_per_track(out_seq)
        # Calculate the absolute difference between the output and target for each shared instrument
        duration_diff = {k: abs(out_duration[k] - tgt_duration[k]) for k in out_duration if k in tgt_duration}
        if len(duration_diff) == 0:
            ret = 0
        else:
            ret = sum(duration_diff.values()) / len(duration_diff)

            # Change the unit to beat
            ret = ret / 12 # unit: beat
        return ret

    def calculate_chroma_iou(self, out_seq, tgt_seq):
        '''
        NOTE: Only works for a bar

        Calculate the IoU for the chroma sequence
        '''
        out_chroma_1, out_chroma_2 = remi_utils.from_remi_bar_get_two_chroma_feat_seq(out_seq)
        tgt_chroma_1, tgt_chroma_2 = remi_utils.from_remi_bar_get_two_chroma_feat_seq(tgt_seq)
        chroma_iou_1 = self.calculate_IoU_for_batched_lists([out_chroma_1], [tgt_chroma_1])[0]
        chroma_iou_2 = self.calculate_IoU_for_batched_lists([out_chroma_2], [tgt_chroma_2])[0]

        # Average the IoU of two chroma features
        chroma_iou = (chroma_iou_1 + chroma_iou_2) / 2

        return chroma_iou

    def calculate_avg_track_pitch_wer(self, out_seq, tgt_seq):
        '''
        Extract track-wise remi sequences from out_seq and tgt_seq
        Compute the sor for all tracks
        Average them.
        '''
        out_track_seqs = from_remi_get_pitch_seq_per_track(out_seq, dedup=True)
        tgt_track_seqs = from_remi_get_pitch_seq_per_track(tgt_seq, dedup=True)
        
        if len(out_track_seqs) == len(tgt_track_seqs) == 0:
            return 1

        overlap_inst = out_track_seqs.keys() & tgt_track_seqs.keys()
        if len(overlap_inst) == 0:
            return 0

        res = []
        for inst in overlap_inst:
            p_seq_out = out_track_seqs[inst]
            p_seq_tgt = tgt_track_seqs[inst]
            inst_wer = self.calculate_wer(p_seq_out, p_seq_tgt)
            res.append(inst_wer)
        mean_wer = sum(res) / len(res)

        return mean_wer

    def calculate_avg_track_pitch_iou(self, out_seq, tgt_seq):
        '''
        Extract track-wise remi sequences from out_seq and tgt_seq
        Compute the sor for all tracks
        Average them.
        '''
        out_track_seqs = from_remi_get_pitch_seq_per_track(out_seq, dedup=True)
        tgt_track_seqs = from_remi_get_pitch_seq_per_track(tgt_seq, dedup=True)
        
        if len(out_track_seqs) == len(tgt_track_seqs) == 0:
            return 1

        overlap_inst = out_track_seqs.keys() & tgt_track_seqs.keys()
        if len(overlap_inst) == 0:
            return 0

        res = []
        for inst in overlap_inst:
            p_seq_out = out_track_seqs[inst]
            p_seq_tgt = tgt_track_seqs[inst]
            inst_iou = self.calculate_IoU_for_batched_lists([p_seq_out], [p_seq_tgt])[0]
            res.append(inst_iou)
        mean_wer = sum(res) / len(res)

        return mean_wer

    def calculate_avg_track_pos_wer(self, out_seq, tgt_seq):
        '''
        Extract track-wise remi sequences from out_seq and tgt_seq
        Compute the sor for all tracks
        Average them.
        '''
        out_track_seqs = remi_utils.from_remi_get_pos_per_track_dict(out_seq, remi_reordered=True)
        tgt_track_seqs = remi_utils.from_remi_get_pos_per_track_dict(tgt_seq, remi_reordered=True)
        
        if len(out_track_seqs) == len(tgt_track_seqs) == 0:
            return 1

        overlap_inst = out_track_seqs.keys() & tgt_track_seqs.keys()
        if len(overlap_inst) == 0:
            return 0

        res = []
        for inst in overlap_inst:
            pos_seq_out = out_track_seqs[inst]
            pos_seq_tgt = tgt_track_seqs[inst]
            pos_wer = self.calculate_wer(pos_seq_out, pos_seq_tgt)
            res.append(pos_wer)
        mean_wer = sum(res) / len(res)

        return mean_wer

    def calculate_avg_track_pos_iou(self, out_seq, tgt_seq):
        '''
        Extract track-wise remi sequences from out_seq and tgt_seq
        Compute the sor for all tracks
        Average them.
        '''
        out_track_seqs = remi_utils.from_remi_get_pos_per_track_dict(out_seq, remi_reordered=True, dedup=True)
        tgt_track_seqs = remi_utils.from_remi_get_pos_per_track_dict(tgt_seq, remi_reordered=True, dedup=True)
        
        if len(out_track_seqs) == len(tgt_track_seqs) == 0:
            return 1

        overlap_inst = out_track_seqs.keys() & tgt_track_seqs.keys()
        if len(overlap_inst) == 0:
            return 0

        res = []
        for inst in overlap_inst:
            pos_seq_out = out_track_seqs[inst]
            pos_seq_tgt = tgt_track_seqs[inst]
            pos_iou = self.calculate_IoU_for_batched_lists([pos_seq_out], [pos_seq_tgt])[0]
            res.append(pos_iou)
        mean_iou = sum(res) / len(res)

        return mean_iou
    
    def calculate_pitch_sor(self, out_seq, tgt_seq):
        '''
        Calculate the pitch sequence sor for two remi sequence
        '''
        out_p_seq = from_remi_get_pitch_seq_global(out_seq)
        tgt_p_seq = from_remi_get_pitch_seq_global(tgt_seq)
        sor = self.calculate_sor(out_p_seq, tgt_p_seq)
        return sor

    def calculate_avg_track_texture_acc(self, out_seq, tgt_seq):
        '''
        Calculate the average texture wer for all tracks
        '''
        if len(tgt_seq) == 1:
            if len(out_seq) == 1:
                return 0
            else:
                return 1

        # Obtain the pos and pitch seq for each track
        out_txt_of_track = self.calculate_texture_for_all_tracks(out_seq)
        tgt_txt_of_track = self.calculate_texture_for_all_tracks(tgt_seq)

        overlap_inst = out_txt_of_track.keys() & tgt_txt_of_track.keys()
        if len(overlap_inst) == 0:
            return 0

        # Calculate accuracy between output and target for each track
        out_txt = []
        tgt_txt = []
        for inst in overlap_inst:
            txt_out = out_txt_of_track[inst]
            out_txt.append(txt_out)
            txt_tgt = tgt_txt_of_track[inst]
            tgt_txt.append(txt_tgt) 

        texture_acc = self.calculate_output_accuracy(txt_out, txt_tgt)

        # res = []
        # for inst in overlap_inst:
        #     txt_out = out_txt_of_track[inst]
        #     txt_tgt = tgt_txt_of_track[inst]
        #     inst_wer = self.calculate_wer(txt_out, txt_tgt)
        #     res.append(inst_wer)
        # mean_wer = sum(res) / len(res)

        return texture_acc
    
    def calculate_texture_for_all_tracks(self, remi_seq):
        '''
        Calculate the texture of each track in the remi sequence
        Return a dict with track as key and texture as value
        '''
        # Obtain the pos and pitch seq for each track
        out_op_seq_of_track = remi_utils.from_remi_get_pos_and_pitch_seq_per_track(remi_seq)
        if 'i-128' in out_op_seq_of_track:
            out_op_seq_of_track.pop('i-128')

        # Determine the texture of each track
        texture_of_track = {}
        for track, op_seq in out_op_seq_of_track.items():
            texture = remi_utils.from_pitch_of_pos_seq_get_texture_of_the_track(op_seq)
            texture_of_track[track] = texture

        return texture_of_track

    def calculate_inst_iou(self, out_seq, tgt_seq):
        '''
        Calculate the IoU for the instrument sequence
        '''
        inst_out = remi_utils.from_remi_get_insts(out_seq)
        inst_tgt = remi_utils.from_remi_get_insts(tgt_seq)
        inst_iou = self.calculate_IoU_for_batched_lists([inst_out], [inst_tgt])
        return inst_iou

    def calculate_inst_iou_from_condition(self, out_seq, condition):
        inst_out = remi_utils.from_remi_get_insts(out_seq, sort_inst=True)
        inst_tgt = [tok for tok in remi_utils.from_condition_get_inst_spec(condition) if tok.startswith('i-')]
        inst_tgt = remi_utils.in_inst_list_sort_inst(inst_tgt)
        inst_iou = self.calculate_IoU_for_batched_lists([inst_out], [inst_tgt])[0]
        return inst_iou

    def calculate_inst_iou_from_inst(self, out_insts, tgt_insts):
        '''
        Calculate the IoU for the instrument sequence
        '''
        inst_iou = self.calculate_IoU_for_batched_lists([out_insts], [tgt_insts])[0]
        return inst_iou

    
    def calculate_groove_wer_sor(self, out_seq, tgt_seq):
        tgt_pos = remi_utils.from_remi_get_pos_seq(tgt_seq)
        out_pos = remi_utils.from_remi_get_pos_seq(out_seq)
        pos_wer = self.calculate_wer(out_pos, tgt_pos)
        pos_sor = self.calculate_sor(out_pos, tgt_pos)
        return pos_wer, pos_sor

    def calculate_groove_wer_sor_mbar(self, out_seq, tgt_seq):
        tgt_pos = remi_utils.from_remi_get_pos_seq_mbar(tgt_seq)
        out_pos = remi_utils.from_remi_get_pos_seq_mbar(out_seq)
        pos_wer = self.calculate_wer(out_pos, tgt_pos)
        pos_sor = self.calculate_sor(out_pos, tgt_pos)
        return pos_wer, pos_sor

    def calculate_groove_iou_mbar(self, out_seq, tgt_seq):
        tgt_pos = remi_utils.from_remi_get_pos_seq_mbar(tgt_seq)
        out_pos = remi_utils.from_remi_get_pos_seq_mbar(out_seq)
        pos_iou = self.calculate_IoU_for_batched_lists([out_pos], [tgt_pos])[0]
        return pos_iou
    

    def calculate_inter_track_div(self, out_seq):
        '''
        Calculate the diversity of the output sequence
        '''
        pitch_seq_per_track = remi_utils.from_remi_get_pitch_seq_per_track(out_seq)

        # Filter out the drum track
        if 'i-128' in pitch_seq_per_track:
            pitch_seq_per_track.pop('i-128')

        # Convert pitch sequence of each track to a 1-d multi-hot vector,
        # Index means pitch number, 1 means have, 0 mean not have
        pitch_dists_per_track = {}
        for inst, pitch_seq in pitch_seq_per_track.items():
            pitch_dist = [0 for i in range(128)]
            for pitch in pitch_seq:
                pitch_id = int(pitch.split('-')[1])
                if pitch_id > 127 or pitch_id < 0:
                    continue
                pitch_dist[pitch_id] = 1
            pitch_dists_per_track[inst] = pitch_dist

        pitch_dists_list = [v for k,v in pitch_dists_per_track.items()]
        if len(pitch_dists_list) == 0:
            return 0, 0

        cos_sim = cosine_similarity(pitch_dists_list, pitch_dists_list)

        # Get the avg and std, of the upper triangular matrix (not include the diagonal)
        sims = []
        for i in range(len(pitch_dists_list)):
            for j in range(i):
                sims.append(cos_sim[i, j])
        
        if len(sims) == 0:
            return 0, 0

        # Get avg and std of sims
        avg_sim = sum(sims) / len(sims)
        std_sim = np.std(sims)
        
        return avg_sim, std_sim

    def calculate_melody_recall_mbar(self, out_seq, tgt_seq):
        res = []
        
        out_bar_indices = remi_utils.from_remi_get_bar_idx(out_seq)
        tgt_bar_indices = remi_utils.from_remi_get_bar_idx(tgt_seq)
        
        for bar_id in [0, 1]:
            if bar_id not in out_bar_indices or bar_id not in tgt_bar_indices:
                continue

            out_bar_start_idx, out_bar_end_idx = out_bar_indices[bar_id]
            tgt_bar_start_idx, tgt_bar_end_idx = tgt_bar_indices[bar_id]
            out_bar_seq = out_seq[out_bar_start_idx:out_bar_end_idx]
            tgt_bar_seq = tgt_seq[tgt_bar_start_idx:tgt_bar_end_idx]

            bar_mel_recall = self.calculate_melody_recall(out_bar_seq, tgt_bar_seq)
            res.append(bar_mel_recall)
        
        if len(res) == 0:
            return 0
        else:
            return sum(res) / len(res)


    def calculate_melody_recall(self, out_seq, tgt_seq):
        tgt_melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            tgt_seq,
            monophonic_only=False,
            top_note=True,
        )
        if len(tgt_melody_seq) > 0:
            # Calculate the melody recall
            tgt_melody_dict = remi_utils.from_melody_pos_pitch_seq_convert_to_dict(tgt_melody_seq)
            out_pos_pitch_seq = remi_utils.from_remi_get_pitch_of_pos_dict(out_seq)
            
            recall_cnt = 0
            for pos in tgt_melody_dict:
                if pos in out_pos_pitch_seq and tgt_melody_dict[pos] in out_pos_pitch_seq[pos]:
                    recall_cnt += 1
            
            recall = recall_cnt / len(tgt_melody_dict)
            return recall
        else:
            return 1
        
    def calculate_melody_f1_q16(self, out_seq, tgt_seq):
        '''
        Calculate the melody F1.
        Definition of melody: notes played by the instrument with the highest average pitch
        '''
        # Locate the 
        tgt_melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            tgt_seq,
            monophonic_only=False,
            top_note=True,
        )
        out_melody_seq = remi_utils.from_remi_get_melody_pos_and_pitch_seq_by_track(
            out_seq,
            monophonic_only=False,
            top_note=True,
        )

        out_proll = get_proll_from_seq_q16(out_melody_seq)
        tgt_proll = get_proll_from_seq_q16(tgt_melody_seq)

        f1 = bar_level_note_f1_from_proll(out_proll, tgt_proll)
        
        return f1

    def calculate_pitch_wer(self, out_seq, tgt_seq):
        # Get the pitch sequence of the output and target
        out_pitch_seq = remi_utils.from_remi_get_pitch_seq_global_mbar(out_seq)
        tgt_pitch_seq = remi_utils.from_remi_get_pitch_seq_global_mbar(tgt_seq)
        pitch_wer = self.calculate_wer(out_pitch_seq, tgt_pitch_seq)

        return pitch_wer

    def calculate_pitch_iou(self, out_seq, tgt_seq):
        # Get the pitch sequence of the output and target
        out_pitch_seq = remi_utils.from_remi_get_pitch_seq_global_mbar(out_seq)
        tgt_pitch_seq = remi_utils.from_remi_get_pitch_seq_global_mbar(tgt_seq)
        pitch_iou = self.calculate_IoU_for_batched_lists([out_pitch_seq], [tgt_pitch_seq])[0]

        return pitch_iou

    def calculate_pitch_wer_per_pos(self, out_seq, tgt_seq):
        # Get the pitch of position dict for output
        out_pos_pitch_dict = {}
        bar_indices = remi_utils.from_remi_get_bar_idx(out_seq)
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_seq = out_seq[bar_start_idx:bar_end_idx]

            out_pos_pitch_dict_bar = remi_utils.from_remi_get_pitch_of_pos_dict(bar_seq)
            out_pos_pitch_dict.update(out_pos_pitch_dict_bar)

        # Get the pitch of position dict for target
        tgt_pos_pitch_dict = {}
        bar_indices = remi_utils.from_remi_get_bar_idx(tgt_seq)
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_seq = tgt_seq[bar_start_idx:bar_end_idx]

            tgt_pos_pitch_dict_bar = remi_utils.from_remi_get_pitch_of_pos_dict(bar_seq)
            tgt_pos_pitch_dict.update(tgt_pos_pitch_dict_bar)

        # Get the shared positions
        shared_pos = set(out_pos_pitch_dict.keys()) & set(tgt_pos_pitch_dict.keys())

        wers = []
        # For each shared position, calculate the WER
        for pos in shared_pos:
            out_pitch = out_pos_pitch_dict[pos]
            tgt_pitch = tgt_pos_pitch_dict[pos]
            wer = self.calculate_wer(out_pitch, tgt_pitch)
            wers.append(wer)

        # Calculate the average WER
        if len(wers) == 0:
            return 0
        else:
            avg_wer = sum(wers) / len(wers)
            return avg_wer

    def calculate_ppl_from_loss(self, loss):
        perplexity = math.exp(loss)
        return perplexity

    def calculate_chord_acc_from_chord(self, out_chord_seq, tgt_chord_seq):
        for i in range(4):
            if out_chord_seq[i] is None:
                out_chord_seq[i] = ('N', 'N')
            if tgt_chord_seq[i] is None:
                tgt_chord_seq[i] = ('N', 'N')
        root_correct_cnt, tgt_correct_cnt = 0, 0
        for out_chord, tgt_chord in zip(out_chord_seq, tgt_chord_seq):
            out_root, out_type = out_chord
            tgt_root, tgt_type = tgt_chord
            
            if out_root == tgt_root:
                root_correct_cnt += 1
            if out_type == tgt_type:
                tgt_correct_cnt += 1

        root_acc = root_correct_cnt / len(out_chord_seq)
        type_acc = tgt_correct_cnt / len(out_chord_seq)
        return root_acc, type_acc

    def calculate_chord_acc_from_chord_two_per_bar(self, out_chord_seq, tgt_chord_seq):
        for i in range(4):
            if out_chord_seq[i] is None:
                out_chord_seq[i] = ('N', 'N')
            if tgt_chord_seq[i] is None:
                tgt_chord_seq[i] = ('N', 'N')
        root_correct_cnt, tgt_correct_cnt = 0, 0

        # Only keep the 1st and 3rd element for two chord seq
        out_chord_seq = [out_chord_seq[0], out_chord_seq[2]]
        tgt_chord_seq = [tgt_chord_seq[0], tgt_chord_seq[2]]

        for out_chord, tgt_chord in zip(out_chord_seq, tgt_chord_seq):
            out_root, out_type = out_chord
            tgt_root, tgt_type = tgt_chord
            
            if out_root == tgt_root:
                root_correct_cnt += 1
            if out_type == tgt_type:
                tgt_correct_cnt += 1

        root_acc = root_correct_cnt / len(out_chord_seq)
        type_acc = tgt_correct_cnt / len(out_chord_seq)
        return root_acc, type_acc
    
    def calculate_note_f1_q16(self, out_seq, tgt_seq):
        '''
        Calculate the note F1 with 16th note resolution
        '''
        out_proll = get_proll_from_seq_q16(out_seq)
        tgt_proll = get_proll_from_seq_q16(tgt_seq)

        f1 = bar_level_note_f1_from_proll(out_proll, tgt_proll)
        return f1
    
    def calculate_note_i_f1_q16(self, out_seq, tgt_seq):
        '''
        Calculate the note F1 with 16th note resolution
        '''
        out_proll = get_proll_from_seq_q16_3d(out_seq)
        tgt_proll = get_proll_from_seq_q16_3d(tgt_seq)

        f1 = bar_level_note_f1_from_proll(out_proll, tgt_proll)
        return f1
    

def get_proll_from_seq_q16(seq):
    '''
    Convert the sequence to a piano roll with 16th note resolution
    '''
    n_pos = 16
    proll = torch.zeros((n_pos, 128), dtype=torch.int)
    
    # Quantize
    pos_q = 0
    for i, tok in enumerate(seq):
        if tok.startswith('o'):
            pos = int(tok.split('-')[1])
            pos_q = round(pos / 3)
        if tok.startswith('p'):
            pitch = int(tok.split('-')[1])

            # Convert drum pitch to normal pitch
            if pitch >= 128:
                pitch = pitch - 128

            # Ensure pos_q and pitch are within the range
            pos_q = min(max(0, pos_q), n_pos - 1)
            pitch = min(max(0, pitch), 127)

            proll[pos_q, pitch] = 1
    return proll

def get_proll_from_seq_q16_3d(seq):
    '''
    Convert the sequence to a piano roll with 16th note resolution
    Shape: (n_inst=128, n_pos=16, pitch=128)
    '''
    n_pos = 16
    proll = torch.zeros((128, n_pos, 128), dtype=torch.int)
    
    # Quantize
    pos_q = 0
    inst = 0
    for i, tok in enumerate(seq):
        if tok.startswith('i'):
            inst = int(tok.split('-')[1])
        if tok.startswith('o'):
            pos = int(tok.split('-')[1])
            pos_q = round(pos / 3)
        if tok.startswith('p'):
            pitch = int(tok.split('-')[1])

            # Ensure pos_q and pitch are within the range
            pos_q = min(max(0, pos_q), n_pos - 1)
            pitch = min(max(0, pitch), 127)
            inst = min(max(0, inst), 127)

            proll[inst, pos_q, pitch] = 1
    return proll


if __name__ == '__main__':
    main()
import os
import sys

sys.path.append('../..')

from src_hf.utils import ls, jpath, create_dir_if_not_exist, update_dic_cnt, save_json, read_json
from utils_midi.utils_midi import RemiTokenizer
from utils_midi import remi_utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import stats
from utils_instrument.general_midi_inst_map import instrument_dict


def main():
    to_remi('REMI_normalized', normalize_pitch=True)


def procedures():
    count_samples()
    to_remi('REMI', normalize_pitch=False)
    count_tokens()
    count_bars()
    count_token_ratio()
    count_inst_dist()
    get_recommended_inst()

    to_remi('REMI_normalized', normalize_pitch=True)


def count_samples():
    '''
    Count number of samples in each split of the SLAKH2100 dataset.
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux'
    res = {}
    splits = ['validation', 'test', 'train']
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        sample_dirnames = ls(split_dir)
        n_samples = len(sample_dirnames)        
        res[split] = n_samples
    tot_samples = sum(res.values())
    res['total'] = tot_samples
    print(res)
    '''
    Result: the current split follows the redux version of the original SLAKH2100 dataset.
    '''


def to_remi(out_dirname, normalize_pitch):
    '''
    Convert the SLAKH2100 dataset to the REMI format.
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux'
    splits = ['validation', 'test', 'train']
    out_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/' + out_dirname
    create_dir_if_not_exist(out_dir)
    tk = RemiTokenizer()
    
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        sample_dirnames = ls(split_dirpath)
        split_out_dirpath = jpath(out_dir, split)
        create_dir_if_not_exist(split_out_dirpath)
        for sample_dirname in tqdm(sample_dirnames):
            sample_dirpath = jpath(split_dirpath, sample_dirname)
            midi_fn = 'all_src.mid'
            midi_fp = jpath(sample_dirpath, midi_fn)
            out_fp = jpath(split_out_dirpath, sample_dirname + '.txt')
            out = tk.midi_to_remi(
                midi_fp, 
                normalize_pitch=normalize_pitch, 
                reorder_by_inst=True
            )
            remi_tok_strs = ' '.join(out)
            with open(out_fp, 'w') as f:
                f.write(remi_tok_strs + '\n')


def count_tokens():
    '''
    Count the number of tokens in the REMI format of the SLAKH2100 dataset.
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI'
    splits = ['validation', 'test', 'train']
    res = {}
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        sample_fns = ls(split_dirpath)
        n_tokens = 0
        for sample_fn in sample_fns:
            sample_fp = jpath(split_dirpath, sample_fn)
            with open(sample_fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    n_tokens += len(line.strip().split())
        res[split] = n_tokens
    print(res)
    '''
    Result: 
    Total tokens: {'validation': 4,378,162, 'test': 2,469,549, 'train': 20,963,029}
    '''

def count_bars():
    '''
    Count the number of tokens in the REMI format of the SLAKH2100 dataset.
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI'
    splits = ['validation', 'test', 'train']
    
    bar_dist = {}
    token_per_bar_dist = {}
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        sample_fns = ls(split_dirpath)
        n_tokens = 0
        for sample_fn in sample_fns:
            sample_fp = jpath(split_dirpath, sample_fn)
            with open(sample_fp, 'r') as f:
                remi_seq = f.read().strip().split(' ')
            bar_indices = remi_utils.from_remi_get_bar_idx(remi_seq)
            bar_cnt = len(bar_indices)
            update_dic_cnt(bar_dist, bar_cnt)

            for bar_id, (bar_start_idx, bar_end_idx) in bar_indices.items():
                bar_seq = remi_seq[bar_start_idx:bar_end_idx]
                token_cnt = len(bar_seq)
                update_dic_cnt(token_per_bar_dist, token_cnt)
        
    save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics'
    # Sort by key
    bar_dist = dict(sorted(bar_dist.items()))
    token_per_bar_dist = dict(sorted(token_per_bar_dist.items()))
    save_json(bar_dist, jpath(save_dir, 'n_bar_distribution.json'))
    save_json(token_per_bar_dist, jpath(save_dir, 'n_token_per_bar_distribution.json'))

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    from scipy import stats

    # Get the quantiles of certain numbers
    # 提取字典中的计数值
    data = []
    for k, v in token_per_bar_dist.items():
        data.extend([k]*v)
    quantiles = [0.5, 0.75, 0.9, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    for q in quantiles:
        print('{}, {}'.format(q, np.percentile(data, q*100)))

    # Visualize the quantiles of both dictionaries
    plt.figure()
    plt.bar(bar_dist.keys(), bar_dist.values())
    plt.xlabel('Number of bars')
    plt.ylabel('Frequency')
    plt.title('Number of bars distribution')
    plt.savefig(jpath(save_dir, 'n_bar_distribution.png'))

    plt.figure()
    plt.bar(token_per_bar_dist.keys(), token_per_bar_dist.values())
    plt.xlabel('Number of tokens per bar')
    plt.ylabel('Frequency')
    plt.title('Number of tokens per bar distribution')
    plt.savefig(jpath(save_dir, 'n_token_per_bar_distribution.png'))


    '''
    Result: 
    Total tokens: {'validation': 4,378,162, 'test': 2,469,549, 'train': 20,963,029}
    '''

def count_token_ratio():
    '''
    Count distribution of token type
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI'
    splits = ['validation', 'test', 'train']
    res = {}
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        sample_fns = ls(split_dirpath)
        n_tokens = 0
        for sample_fn in sample_fns:
            sample_fp = jpath(split_dirpath, sample_fn)
            with open(sample_fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.strip().split()
                    for token in tokens:
                        token_type = token.split('-')[0]
                        update_dic_cnt(res, token_type)
    # Convert value to percentage
    total_tokens = sum(res.values())
    for k in res:
        res[k] = round(res[k] / total_tokens, 4)
    for k in res:
        print('{}: {}%'.format(k, res[k]*100))
    '''
    Result: 
    s: 0.65%
    t: 0.65%
    b: 0.65%
    i: 3.75%
    o: 21.12%
    p: 36.59%
    d: 36.59%
    '''

def count_inst_dist():
    '''
    Count instrument distribution
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/REMI'
    splits = ['validation', 'test', 'train']
    res = {}
    for split in splits:
        split_dirpath = jpath(dataset_dir, split)
        sample_fns = ls(split_dirpath)
        n_tokens = 0
        for sample_fn in sample_fns:
            sample_fp = jpath(split_dirpath, sample_fn)
            with open(sample_fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.strip().split()
                    for token in tokens:
                        if token.startswith('i'):
                            update_dic_cnt(res, token)
    
    # # Convert value to percentage
    # total_tokens = sum(res.values())
    # for k in res:
    #     res[k] = round(res[k] / total_tokens, 4)
    # for k in res:
    #     print('{}: {}%'.format(k, res[k]*100))

    # Sort res by key's id
    res = dict(sorted(res.items(), key=lambda x: int(x[0].split('-')[-1]), reverse=False))

    save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics'
    save_json(res, jpath(save_dir, 'instrument_distribution.json'))

    # Draw a bar chart
    
    plt.figure(figsize=(10, 15))  # 增加图形的高度，为每个标签提供足够的空间
    plt.barh(list(res.keys()), list(res.values()))  # 使用barh创建水平条形图
    plt.ylabel('Instrument tokens')  # 这将成为 y 轴标签
    plt.xlabel('Frequency')  # 这将成为 x 轴标签
    plt.title('Number of bars distribution')
    plt.tight_layout()  # 优化布局

    # 可能需要调整标签位置或图形边缘以确保所有标签都可见
    plt.subplots_adjust(left=0.2)  # 根据需要调整，保证左侧标签不被剪切

    plt.savefig(jpath(save_dir, 'Inst_distribution.png'))

def get_recommended_inst():
    '''
    Find out the instrument tokens that are not recommended.
    '''
    save_dir = '/home/longshen/work/musecoco/dataset_preparation/slakh/statistics'
    inst_dist = read_json(jpath(save_dir, 'instrument_distribution.json'))
    
    # 提取所有出现次数
    frequencies = list(inst_dist.values())

    # 计算第 10 百分位数
    quantiles = {}
    qs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for q in qs:
        v = np.percentile(frequencies, q)
        quantiles[q] = v
    save_json(quantiles, jpath(save_dir, 'tokens_per_inst_quantiles.json'))

    # Sort inst_dist by value, reverse
    inst_dist = dict(sorted(inst_dist.items(), key=lambda x: x[1], reverse=True))

    # Draw a bar chart from inst_dist
    plt.figure(figsize=(10, 15))  # 增加图形的高度，为每个标签提供足够的空间
    plt.barh(list(inst_dist.keys()), list(inst_dist.values()))  # 使用barh创建水平条形图
    plt.ylabel('Instrument tokens')  # 这将成为 y 轴标签
    plt.xlabel('Frequency')  # 这将成为 x 轴标签
    plt.savefig(jpath(save_dir, 'token_per_inst.png'))
    
    ''' Get the recommended instrument (>=80%) '''
    q = np.percentile(frequencies, 80)
    print(q)

    # Get the inst token that is has least 10% frequency, or not appeared in the dict
    recommended_insts = {}
    all_insts = ['i-{}'.format(j) for j in range(0, 129)]
    for inst in all_insts:
        if inst in inst_dist and inst_dist[inst] >= q:
            recommended_insts[inst] = inst_dist[inst]
    recommended_insts = dict(sorted(recommended_insts.items(), key=lambda x: int(x[0].split('-')[-1]), reverse=False))

    # Add inst name to the dict
    for inst in recommended_insts:
        inst_name = instrument_dict[int(inst.split('-')[-1])]
        recommended_insts[inst] = 'freq: {}, name: {}'.format(recommended_insts[inst], inst_name)

    save_json(recommended_insts, jpath(save_dir, 'inst_manageable.json'))

    ''' Get the challenging instruments (40% <= x < 80% ) '''
    q1 = np.percentile(frequencies, 40)
    q2 = np.percentile(frequencies, 80)
    challenging_insts = {}
    for inst in all_insts:
        if inst in inst_dist and inst_dist[inst] >= q1 and inst_dist[inst] < q2:
            challenging_insts[inst] = inst_dist[inst]
    challenging_insts = dict(sorted(challenging_insts.items(), key=lambda x: int(x[0].split('-')[-1]), reverse=False))
    # Add inst name to the dict
    for inst in challenging_insts:
        inst_name = instrument_dict[int(inst.split('-')[-1])]
        challenging_insts[inst] = 'freq: {}, name: {}'.format(challenging_insts[inst], inst_name)
    save_json(challenging_insts, jpath(save_dir, 'inst_challenging.json'))

    ''' Get "tough" instruments (< 40%) '''
    q = np.percentile(frequencies, 40)
    tough_insts = {}
    for inst in all_insts:
        if inst in inst_dist and inst_dist[inst] < q:
            tough_insts[inst] = inst_dist[inst]
    tough_insts = dict(sorted(tough_insts.items(), key=lambda x: int(x[0].split('-')[-1]), reverse=False))
    # Add inst name to the dict
    for inst in tough_insts:
        inst_name = instrument_dict[int(inst.split('-')[-1])]
        tough_insts[inst] = 'freq: {}, name: {}'.format(tough_insts[inst], inst_name)
    save_json(tough_insts, jpath(save_dir, 'inst_tough.json'))



if __name__ == '__main__':
    main()
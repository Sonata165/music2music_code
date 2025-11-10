import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(os.path.abspath(__file__))))

from utils_common.utils import read_json, jpath, save_json
import math


def main():
    # count_tokens_per_note()
    # calculate_avg_token_cnt_per_bar()
    calculate_shannon_entropy2()


def procedures():
    count_token_distribution_remiz()
    count_token_distribution_remiplus()
    calculate_avg_token_cnt_per_bar()
    convert_token_cnt_to_token_freq()
    calculate_shannon_entropy()
    count_tokens_per_note()


def count_tokens_per_note():
    '''
    Count the number of tokens per note in the dataset
    '''
    data_fps = [
        '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json',
        '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_remiplus.json'
    ]
    for data_fp in data_fps:
        data = read_json(data_fp)
        n_token_per_note = []
        for split in data:
            split_data = data[split]
            for bar_name in split_data:
                seq = split_data[bar_name]['content'].strip().split()[2:]
                seq_str = ' '.join(seq)
                n_notes = seq_str.count('p')
                if n_notes == 0:
                    continue
                n_toks = len(seq)
                n_toks_per_note = n_toks / n_notes
                n_token_per_note.append(n_toks_per_note)
            
        print(data_fp)
        print('Average token count per note:', sum(n_token_per_note) / len(n_token_per_note))


def calculate_shannon_entropy():
    '''
    Calculate the Shannon entropy of the token distribution
    '''
    data_fps = [
        '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json',
        '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_remiplus.json'
    ]
    freq_fps = [
        '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results/token_cnt_remiz_freq.json',
        '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results/token_cnt_remiplus_freq.json'
    ]
    for data_fp, freq_fp in zip(data_fps, freq_fps):
        data = read_json(data_fp)
        freq = read_json(freq_fp)
        entropies = []
        for split in data:
            split_data = data[split]
            for bar_name in split_data:
                bar = split_data[bar_name]['content'].strip().split()
                bar_entropy = 0
                for token in bar:
                    bar_entropy -= freq[token] * math.log2(freq[token])
                entropies.append(bar_entropy)

        print(data_fp)
        print('Shannon entropy:', sum(entropies) / len(entropies))
        # print('Shannon entropy:', entropy)


def calculate_shannon_entropy2():
    '''
    Calculate the Shannon entropy of the token distribution
    P comes from each single bar sequence
    '''
    data_fps = [
        '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json',
        '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_remiplus.json'
    ]
    for data_fp in data_fps:
        data = read_json(data_fp)
        entropies = []
        for split in data:
            split_data = data[split]
            for bar_name in split_data:
                bar = split_data[bar_name]['content'].strip().split()

                # Calculate the frequency of each token
                freq = {}
                for token in bar:
                    if token not in freq:
                        freq[token] = 0
                    freq[token] += 1
                for token in freq:
                    freq[token] /= len(bar)

                bar_entropy = 0
                for token in bar:
                    bar_entropy -= freq[token] * math.log2(freq[token])
                entropies.append(bar_entropy)

        print(data_fp)
        print('Shannon entropy:', sum(entropies) / len(entropies))
        # print('Shannon entropy:', entropy)


def convert_token_cnt_to_token_freq():
    '''
    Convert token count to token frequency
    '''
    data_fps = [
        '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results/token_cnt_remiz.json',
        '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results/token_cnt_remiplus.json'
    ]
    for data_fp in data_fps:
        data = read_json(data_fp)
        total_cnt = sum(data.values())
        token_freq = {token: cnt / total_cnt for token, cnt in data.items()}

        data_name = data_fp.split('/')[-1].split('.')[0]
        save_json(token_freq, jpath('/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results', f'{data_name}_freq.json'))


def calculate_avg_token_cnt_per_bar():
    '''
    Calculate the average token count per bar
    '''
    data_fps = [
        '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results/bar_len_cnt_remiz.json',
        '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results/bar_len_cnt_remiplus.json'
    ]
    for data_fp in data_fps:
        data = read_json(data_fp)
        num = [i - 2 for i in data]
        avg_len = sum(num) / len(data)

        data_name = data_fp.split('/')[-1].split('.')[0]
        print(data_name)
        print('Average token count per bar:', avg_len)


def count_token_distribution_remiplus():
    '''
    Count the distribution of tokens in the dataset
    For REMI+ tokenizer
    '''
    data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_remiplus.json'
    result_dir = '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results'
    data = read_json(data_fp)
    token_cnt = {}
    bar_len_cnt = []
    for split in data:
        split_data = data[split]
        for bar_name in split_data:
            bar = split_data[bar_name]['content'].strip().split()
            bar_len = len(bar)
            bar_len_cnt.append(bar_len)
            for token in bar:
                if token not in token_cnt:
                    token_cnt[token] = 0
                token_cnt[token] += 1

    save_json(token_cnt, jpath(result_dir, 'token_cnt_remiplus.json'))
    save_json(bar_len_cnt, jpath(result_dir, 'bar_len_cnt_remiplus.json'))


def count_token_distribution_remiz():
    '''
    Count the distribution of tokens in the dataset
    For REMI-z tokenizer
    '''
    data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json'
    result_dir = '/home/longshen/work/MuseCoco/musecoco/tokenizer_effeciency/results'
    data = read_json(data_fp)
    token_cnt = {}
    bar_len_cnt = []
    for split in data:
        split_data = data[split]
        for bar_name in split_data:
            bar = split_data[bar_name]['content'].strip().split()
            bar_len = len(bar)
            bar_len_cnt.append(bar_len)
            for token in bar:
                if token not in token_cnt:
                    token_cnt[token] = 0
                token_cnt[token] += 1

    save_json(token_cnt, jpath(result_dir, 'token_cnt_remiz.json'))
    save_json(bar_len_cnt, jpath(result_dir, 'bar_len_cnt_remiz.json'))


if __name__ == '__main__':
    main()
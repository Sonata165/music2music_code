import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(os.path.abspath(__file__))))

from utils_common.utils import read_json
from remi_z import MultiTrack, Bar

def main():
    data_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/metadata/segment_dataset_1bar_norm_withhist.json'
    data = read_json(data_fp)

    sparse_ratio = []
    for split in data:
        data_split = data[split]
        for bar_name, bar in data_split.items():
            # Count notes by pitch token
            bar_str = bar['content']
            n_notes = bar_str.count('p')

            ts = bar['meta']['time_signature']
            insts = bar['meta']['insts'].strip().split(' ')
            n_insts = len(insts)
            n_beats = int(ts[1])
            tot_pos = n_beats * 48 / 3
            tot_notes = 128

            denom = tot_pos * tot_notes * n_insts
            if denom > 0:

                sparsity = n_notes / (tot_pos * tot_notes * n_insts)
                sparse_ratio.append(sparsity)

    print('Mean sparsity:', sum(sparse_ratio) / len(sparse_ratio))


if __name__ == '__main__':
    main()
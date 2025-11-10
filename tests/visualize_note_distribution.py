import os
import sys

sys.path.append('..')

from src_hf.utils import read_json

def main():
    data_fp = '/data2/longshen/musecoco_data/data_statistics/notes_of_insts.json'
    save_dir = os.path.dirname(data_fp)

    data = read_json(data_fp)

    import matplotlib.pyplot as plt

    keys = list(data.keys())
    values = [item[0] for item in data.values()]
    plt.bar(keys, values)
    plt.xlabel('Key')
    plt.ylabel('Value')
    plt.title('Notes per instruments')
    plt.xticks(rotation=270)
    plt.subplots_adjust(bottom=0.3)  # Adjust the bottom margin
    save_fp = os.path.join(save_dir, 'note_distribution_with_drum.png')
    plt.savefig(save_fp)

    data.pop('drums')

    keys = list(data.keys())
    values = [item[0] for item in data.values()]
    plt.figure()
    plt.bar(keys, values)
    plt.xlabel('Key')
    plt.ylabel('Value')
    plt.title('Notes per instruments')
    plt.xticks(rotation=270)
    plt.subplots_adjust(bottom=0.3)  # Adjust the bottom margin
    save_fp = os.path.join(save_dir, 'note_distribution_no_drum.png')
    plt.savefig(save_fp)


if __name__ == '__main__':
    main()
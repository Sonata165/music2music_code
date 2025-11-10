import os
import json
import numpy as np
import yaml
import re

jpath = os.path.join
pexist = os.path.exists


def ls(path):
    '''
    List all files and folders in the given path
    Remove ".DS_Store"
    '''
    fnames = os.listdir(path)
    if '.DS_Store' in fnames:
        fnames.remove('.DS_Store')
    fnames.sort()
    return fnames


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def save_json(data, path, sort=False, indent=4, ):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=indent, sort_keys=sort, ensure_ascii=False))


def print_json(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))


def read_yaml(fp):
    '''
    This function seems to be very slow when loading large files
    '''
    with open(fp, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data, fp):
    with open(fp, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def print_yaml(data):
    print(yaml.safe_dump(data))


def my_log(log_fn, log_str):
    with open(log_fn, 'a') as f:
        f.write(log_str)


def dict_of_lists_to_list_of_dicts(d):
    # Check if the dictionary is empty
    if not d:
        return []

    # Get the length of the lists in the dictionary
    try:
        length = len(next(iter(d.values())))
    except Exception as e:
        raise ValueError("Error in determining the length of lists.") from e

    # Check if all lists in the dictionary are of the same length
    if not all(len(lst) == length for lst in d.values()):
        raise ValueError("All lists in the dictionary must have the same length.")

    # Create a list of dictionaries
    list_of_dicts = []
    for i in range(length):
        new_dict = {key: d[key][i] for key in d}
        list_of_dicts.append(new_dict)

    return list_of_dicts


class ProgramIdUtil:
    def __init__(self):
        fp = jpath(os.path.dirname(__file__), 'datasets/inst_def/slakh_inst_rev.json')
        self.inst_map = read_json(fp)

    def midi_program_to_slakh_program(self, pid):
        if pid == -1:
            ret = 0  # Drum
        else:
            ret = int(self.inst_map[str(pid)]['inst_type_id'])
        return ret


def get_hostname():
    '''
    Get the name of a computer
    '''
    import socket
    server_name = socket.gethostname().strip()
    return server_name


def get_dataset_loc_by_hostname(hostname):
    '''
    Return the path of dataset, for different servers
    '''
    cur_dir = os.path.dirname(__file__)
    dic_fp = jpath(cur_dir, 'misc/slakh_h5_path.yaml')
    h5_fp_dic = read_yaml(dic_fp)
    ret = h5_fp_dic[hostname]
    return ret


def get_dataset_loc():
    '''
    Get the corresponding h5 location
    '''
    host_name = get_hostname()
    loc = get_dataset_loc_by_hostname(host_name)
    return loc


def get_dataset_dir():
    '''
    Get the path to the dataset folder
    '''
    hostname = get_hostname()
    # cur_dir = os.path.dirname(__file__)
    # dic_fp = jpath(cur_dir, 'misc/slakh_h5_path.yaml')
    dic_fp = '/home/longshen/work/StyleTrans/modules/musecoco/misc/slakh_h5_path.yaml'
    h5_fp_dic = read_yaml(dic_fp)
    ret = h5_fp_dic['dir'][hostname]
    return ret


def sort_dict_by_key(dic, reverse=False):
    t = list(dic.items())
    t.sort(reverse=reverse)
    ret = dict(t)
    return ret


def accumulate_dic(dic, k):
    if k in dic:
        dic[k] += 1
    else:
        dic[k] = 1


def update_dic(dic, k, v):
    if k in dic:
        dic[k].append(v)
    else:
        dic[k] = [v]

def update_dic_cnt(dic, k):
    if k in dic:
        dic[k] += 1
    else:
        dic[k] = 1


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        # os.mkdir(dir)
        os.makedirs(dir)


def save_remi(remi_seq, fp):
    remi_str = ' '.join(remi_seq)
    with open(fp, 'w') as f:
        f.write(remi_str + '\n')


def read_remi(fp, split=True):
    with open(fp) as f:
        remi_str = f.readline().strip()
    if split:
        remi_seq = remi_str.split(' ')
    else:
        remi_seq = remi_str
    return remi_seq


class ChordUtil:
    def __init__(self):
        '''

        '''
        self.junyan_to_magenta_dict = {
            'min/b7': 'm',
            'min/2': 'm',
            'maj/b7': 'maj',
            'maj/2': 'maj',
            'sus4(b7)': 'maj',
            'sus2': 'maj',
            'sus4': 'maj',
            '13': '7',
            '11': '7',
            'min9': 'm7',
            '9': '7',
            'maj9': 'maj7',
            'dim7': 'dim',
            'hdim7': 'm7b5',
            'min7': 'm7',
            '7': '7',
            'maj7': 'maj7',
            'min/5': 'm',
            'min/b3': 'm',
            'maj/5': 'maj',
            'maj/3': 'maj',
            'dim': 'dim',
            'aug': 'aug',
            'min': 'm',
            'maj': 'maj',
            'N': 'N',
        }

        self.chord_qualities = {  # Junyan's MIDI chord recognizer
            'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'aug': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            'sus4(b7)': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            'sus4(b7,9)': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            'sus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            '7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            'maj6': [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            'min6': [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            '9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'maj9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'min9': [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            '7(#9)': [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            'maj(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'min(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'maj(11)': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            '11': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            'maj9(11)': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            'min11': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            '13': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            'maj13': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            'min13': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'dim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'hdim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            # '5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        }

    def convert_chord_type_from_junyan_to_meganta(self, chord_type):
        if chord_type not in self.junyan_to_magenta_dict:
            print('chord type error: |{}|'.format(chord_type))
            exit(20)
        ret = self.junyan_to_magenta_dict[chord_type]
        return ret

    def normalize_chord_vocab_for_chord_seq(self, chord_seq):
        '''
        For a sequence of chords, change each of the chord to meganta type
        :param chord_seq: a list of str, chord sequence
        '''
        ret = []

        for chord in chord_seq:
            # print(chord)
            t = chord.strip().split(':')
            root = t[0]
            type = t[-1]
            # print('root {}, type {}'.format(root, type))
            type_meganta = self.convert_chord_type_from_junyan_to_meganta(type)
            chord_new = '{}:{}'.format(root, type_meganta)
            ret.append(chord_new)
        return ret


def transpose_chord(chord, pitch_shift):
    # 定义和弦根音到MIDI值的映射
    notes_to_midi = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    # MIDI值到和弦根音的映射
    midi_to_notes = {v: k for k, v in notes_to_midi.items()}

    # 如果和弦为"N"，直接返回
    if chord == "N":
        return "N"

    # 提取和弦的根音和类型
    root, chord_type = chord.split(':')

    # 计算新的根音
    if root in notes_to_midi:  # 确保根音有效
        original_midi = notes_to_midi[root]
        new_midi = (original_midi + pitch_shift) % 12  # 考虑循环
        new_root = midi_to_notes[new_midi]
    else:
        return "Invalid chord"  # 如果根音无效，返回错误

    # 返回转调后的和弦
    return f"{new_root}:{chord_type}"

def get_latest_checkpoint(base_dir):
    # 构建lightning_logs的路径
    logs_dir = os.path.join(base_dir, 'lightning_logs')
    
    # 确保该目录存在
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"The directory {logs_dir} does not exist.")

    # 查找所有的version_X文件夹，获取最大的版本号
    versions = [d for d in os.listdir(logs_dir) if re.match(r'^version_\d+$', d)]
    if not versions:
        raise ValueError("No version directories found in lightning_logs.")
    
    # 获取最大的版本号
    latest_version = max(versions, key=lambda v: int(v.split('_')[1]))
    latest_version_dir = os.path.join(logs_dir, latest_version, 'checkpoints')

    # 确保checkpoints目录存在
    if not os.path.exists(latest_version_dir):
        raise FileNotFoundError(f"No checkpoints directory found in {latest_version_dir}")

    # 检查checkpoints目录下的文件
    checkpoints = os.listdir(latest_version_dir)
    if len(checkpoints) != 1:
        raise AssertionError("There should be exactly one checkpoint file in the directory.")
    
    # 获取checkpoint文件的完整路径
    checkpoint_path = os.path.join(latest_version_dir, checkpoints[0])
    return latest_version, checkpoint_path
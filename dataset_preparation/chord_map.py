import os
import sys

from utils import jpath


def _main():
    pass
    test_chord_tokenizer()


def test_chord_tokenizer():
    import numpy as np
    t = np.array([
        ['N', 'N', 'N', 'N', 'D:min', 'D:min', 'D:min', 'D:min'],
        ['D:min', 'D:min', 'D:min', 'D:min', 'D:min', 'G:7', 'G:7', 'G:7']
    ])
    tk = ChordTokenizer()
    tk.tokenize_chord_list(t[0])


def _procedures():
    pass


def chord_type_to_id():
    pass


class ChordTokenizer:
    def __init__(self):
        # Read the chord type definition file, construct chord type dictionary
        type_def_fn = 'junyan_midi.txt'  # 'submission_chord_list.txt'
        type_def_fp = jpath(os.path.dirname(__file__), 'chord_def', type_def_fn)
        with open(type_def_fp) as f:
            chord_types = f.readlines()
        chord_types = [i.strip().split(':')[-1] for i in chord_types]
        self.type_to_id = {}
        self.id_to_type = {}
        cnt = 0
        for chord_type in chord_types:
            self.type_to_id[chord_type] = cnt
            self.id_to_type[cnt] = chord_type
            cnt += 1

        # Construct chord root dictionary
        self.root_to_id = {
                'N': 0, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
                'E': 5, 'Fb': 5, 'E#': 6, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9,
                'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12,
            }
        self.id_to_root = {}
        for root in self.root_to_id:
            self.id_to_root[id] = root

    # def __init__(self):
    #     # Read the chord type definition file, construct chord type dictionary
    #     type_def_fn = 'junyan_midi.txt'  # 'submission_chord_list.txt'
    #     type_def_fp = jpath(os.path.dirname(__file__), 'chord_def', type_def_fn)
    #     with open(type_def_fp) as f:
    #         chord_types = f.readlines()
    #     chord_types = [i.strip().split(':')[-1] for i in chord_types]
    #     self.type_to_id = {}
    #     self.id_to_type = {}
    #     cnt = 0
    #     for chord_type in chord_types:
    #         self.type_to_id[chord_type] = cnt
    #         self.id_to_type[cnt] = chord_type
    #         cnt += 1
    #
    #     # Construct chord root dictionary
    #     note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    #     roots = []
    #     for note_name in note_names:
    #         roots.append(note_name + 'b')
    #         roots.append(note_name)
    #     roots.insert(0, 'N')
    #     # roots.append('N')
    #     cnt = 0
    #     self.root_to_id = {}
    #     self.id_to_root = {}
    #     for root in roots:
    #         self.root_to_id[root] = cnt
    #         self.id_to_root[cnt] = root
    #         cnt += 1

    def chord_type_to_type_id(self, chord_type):
        return self.type_to_id[chord_type]

    def type_id_to_chord_type(self, type_id):
        return self.id_to_type[type_id]

    def chord_root_to_root_id(self, chord_root):
        return self.root_to_id[chord_root]

    def root_id_to_chord_root(self, root_id):
        return self.id_to_root[root_id]

    def tokenize_chord_list(self, chord_list):
        '''
        Convert a list of chord symbols to two output lists
        :return: A list of root ids, and a list of type ids.
        '''
        if isinstance(chord_list[0], bytes):
            chord_list = [i.decode("utf-8") for i in chord_list]
        t = [i.split(':') for i in chord_list]
        chord_roots = [i[0] for i in t]
        chord_types = [i[-1].split('/')[0] for i in t]  # remove inversion info
        ret = []
        for chord_root, chord_type in zip(chord_roots, chord_types):
            root_id = self.chord_root_to_root_id(chord_root)
            type_id = self.chord_type_to_type_id(chord_type)
            root_token = 'CR-{}'.format(root_id)
            type_token = 'CT-{}'.format(type_id)
            ret.append(root_token)
            ret.append(type_token)
        return ret

    def tokenize_chord_list_seprate(self, chord_list):
        '''
        Convert a list of chord symbols to two output lists
        :return: A list of root ids, and a list of type ids.
        '''
        if isinstance(chord_list[0], bytes):
            chord_list = [i.decode("utf-8") for i in chord_list]
        t = [i.split(':') for i in chord_list]
        chord_roots = [i[0] for i in t]
        chord_types = [i[-1].split('/')[0] for i in t]  # remove inversion info
        crs = []
        cts = []
        for chord_root, chord_type in zip(chord_roots, chord_types):
            root_id = self.chord_root_to_root_id(chord_root)
            type_id = self.chord_type_to_type_id(chord_type)
            root_token = 'CR-{}'.format(root_id)
            type_token = 'CT-{}'.format(type_id)
            crs.append(root_token)
            cts.append(type_token)
        return crs, cts




if __name__ == '__main__':
    _main()

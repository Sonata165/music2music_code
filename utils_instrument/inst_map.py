import os
import sys

from utils_common.utils import read_json, print_json, save_yaml, read_yaml, jpath


def _main():
    test_map_util()


def test_map_util():
    id = 27
    map_util = InstMapUtil()
    print(map_util.slakh_from_midi_program_get_id_and_inst(id))


def generate_id_to_inst_map_slakh():
    '''
    Generate a file containing the Slakh ID to instrument name mapping
    :return:
    '''
    def_fp = './inst_def/slakh_inst_rev.json'
    inst_def = read_json(def_fp)
    ret = {}
    for k in inst_def:
        midi_program = k
        entry = inst_def[midi_program]
        inst_type = entry['instrument_type'].lower().replace(' ', '_').replace('/', '_')
        slakh_id = entry['inst_type_id']
        if slakh_id not in ret:
            ret[slakh_id] = inst_type
    print_json(ret)

    out_fp = './inst_def/slakh_id_to_inst_mapping.yaml'
    save_yaml(ret, out_fp)


class InstMapUtil:
    def __init__(self):
        cur_dir = os.path.dirname(__file__)

        # Slakh related
        slakh_def_fp = jpath(cur_dir, 'inst_def/slakh_inst_rev.json')
        self.inst_def = read_json(slakh_def_fp)
        slakh_id_to_inst_fp = jpath(cur_dir, 'inst_def/slakh_id_to_inst_mapping.yaml')
        self.slakh_id_to_inst = read_yaml(slakh_id_to_inst_fp)
        self.slakh_inst_to_id = {}
        for id in self.slakh_id_to_inst:
            inst = self.slakh_id_to_inst[id]
            self.slakh_inst_to_id[inst] = int(id)

        # slakh id to ma2s inst mapping
        fp = jpath(cur_dir, 'inst_def/slakh_id_to_ma2s_inst.yaml')
        self.slakh_id_to_ma2s_inst_dic = read_yaml(fp)
        fp = jpath(cur_dir, 'inst_def/ma2s_id_to_inst_mapping.yaml')
        self.ma2s_id_to_inst = read_yaml(fp)
        self._ma2s_inst_to_id = {}
        for k in self.ma2s_id_to_inst:
            v = self.ma2s_id_to_inst[k]
            self._ma2s_inst_to_id[v] = int(k)

        # Dict for instrument quantization
        seen_inst = {}
        self.first_prog_of_inst = {}
        for prog in self.inst_def:
            inst_name = self.inst_def[prog]['instrument_type']
            if inst_name not in seen_inst:
                self.first_prog_of_inst[prog] = prog
                seen_inst[inst_name] = prog
            else:
                self.first_prog_of_inst[prog] = seen_inst[inst_name]

        # Reverse self.first_prog_of_inst
        self.slakh_supported_prog_ids = set([v for k, v in self.first_prog_of_inst.items()])


    def slakh_get_supported_prog_ids(self):
        return self.slakh_supported_prog_ids


    def ma2s_get_all_inst_id(self):
        ret = []
        for id in self.ma2s_id_to_inst:
            ret.append(int(id))
        return ret

    def ma2s_get_all_inst_name(self):
        ret = []
        for id in self.ma2s_id_to_inst:
            ret.append(self.ma2s_id_to_inst[id])
        return ret

    def slakh_get_all_inst_name(self):
        '''
        Return a list of instrument names that contains all possible instruments in Slakh dataset
        In the order of slakh id from small to big
        '''
        ret = []
        for id in self.slakh_id_to_inst:
            ret.append(self.slakh_id_to_inst[id])
        return ret

    def slakh_get_all_inst_id(self):
        ret = []
        for id in self.slakh_id_to_inst:
            ret.append(int(id))
        return ret

    def slakh_get_inst_name_from_prettymidi_inst(self, inst):
        is_drum = inst.is_drum
        program_id = inst.program
        slakh_id, inst_name = self.slakh_from_midi_program_get_id_and_inst(program_id, is_drum)
        return inst_name

    def slakh_from_midi_program_get_id_and_inst(self, program_id, is_drum=False):
        '''
        program id is [0, 127] according to General MIDI, and one additional 128 indicating drum track
        :param program_id:
        :return:
        '''
        if is_drum:
            program_id = 128
        else:
            pass
        entry = self.inst_def[str(program_id)]
        inst_type = entry['instrument_type'].lower().replace(' ', '_').replace('/', '_')
        slakh_id = entry['inst_type_id']
        return slakh_id, inst_type
    
    def slakh_from_midi_program_get_inst_id(self, program_id: int) -> int:
        entry = self.inst_def[str(program_id)]
        slakh_id = int(entry['inst_type_id'])
        return slakh_id

    def ma2s_get_id_and_inst_from_midi_program(self, program_id, is_drum=False):
        '''
        program id is [0, 127] according to General MIDI, and one additional 128 indicating drum track
        :param program_id:
        :return:
        '''
        slakh_id, inst_type = self.slakh_from_midi_program_get_id_and_inst(program_id, is_drum)
        id, inst = self.ma2s_get_id_and_inst_from_slakh_id(slakh_id)
        return id, inst

    def ma2s_get_id_and_inst_from_slakh_id(self, slakh_id):
        inst = self.slakh_id_to_ma2s_inst_dic[str(slakh_id)]
        id = self._ma2s_inst_to_id[inst]
        return id, inst

    def ma2s_inst_name_to_id(self, inst_name):
        ret = self._ma2s_inst_to_id[inst_name]
        return ret

    def slakh_id_to_inst_name(self, id):
        ret = self.slakh_id_to_inst[str(id)]
        return ret

    def slakh_inst_name_to_id(self, inst_name):
        ret = self.slakh_inst_to_id[inst_name]
        return ret

    def slakh_quantize_inst_prog(self, midi_prog: int) -> int:
        '''
        Return the first program number within the same category as the midi_prog
        E.g., instrument prog=4 is a
        '''
        midi_prog = str(midi_prog)

        if self.slack_support_instrument(midi_prog):
            ret = self.first_prog_of_inst[midi_prog]
            ret = int(ret)
        else:
            # raise Exception, 'Instrument not supported'
            # print('WARNING: Instrument not supported')
            ret = None

        return ret

    def slack_support_instrument(self, midi_prog):
        if midi_prog in self.inst_def:
            return True
        else:
            return False


if __name__ == '__main__':
    _main()

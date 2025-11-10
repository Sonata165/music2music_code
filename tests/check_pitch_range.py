'''
Track00674 has pitch=-1, why?
'''

from remi_z import MultiTrack, Bar

midi_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/original/train/Track00674/all_src.mid'
mt = MultiTrack.from_midi(midi_fp)
print(mt.get_pitch_range(return_range=True))

mt.normalize_pitch()
print(mt.get_pitch_range(return_range=True))

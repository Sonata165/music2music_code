'''
Why there is dur=127 in Track01876?
'''

from remi_z import MultiTrack, Bar

midi_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/original/test/Track01876/all_src.mid'
mt = MultiTrack.from_midi(midi_fp)
out_fp = '/home/longshen/work/MuseCoco/musecoco/_misc/Track01876_dec.mid'
mt.to_midi(out_fp)

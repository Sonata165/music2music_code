import os
import sys
sys.path.append('/home/longshen/work/Music2music/music2music')
from api.arranger import BandArranger, PianoArranger, DrumArranger


def main():
    # test_band_arranger()
    # test_piano_arranger()
    test_drum_arranger()


def test_band_arranger():
    model = BandArranger('LongshenOu/m2m_arranger', hf_ckpt=True)
    song_fp = '/data2/longshen/musecoco_data/full_song/caihong/caihong.mid'
    
    # # Using preset
    # preset = 'string_trio' # 'rock_band' 'jazz_band'
    # arranged_mt = model.arrange(song_fp, use_preset=preset)
    # print(arranged_mt)
    # save_fp = f'/data2/longshen/musecoco_data/infer_out/test_api/arranger/caihong_{preset}.mid'
    # arranged_mt.to_midi(save_fp)

    # Using custom instruments
    # inst = [80, 16] # synth lead, synth pad
    inst = [80, 8] # synth lead, e-piano
    arranged_mt2 = model.arrange(song_fp, use_preset=None, instrument_and_voice=inst)
    print(arranged_mt2)
    save_fp2 = f'/data2/longshen/musecoco_data/infer_out/test_api/arranger/caihong_custom.mid'
    arranged_mt2.to_midi(save_fp2)


def test_piano_arranger():
    model = PianoArranger('LongshenOu/m2m_pianist_dur', hf_ckpt=True)
    song_fp = '/data2/longshen/musecoco_data/full_song/caihong/caihong.mid'
    
    # Using preset
    preset = 'piano' # The only option
    arranged_mt = model.arrange(song_fp, use_preset=preset)
    print(arranged_mt)
    save_fp = f'/data2/longshen/musecoco_data/infer_out/test_api/arranger/caihong_{preset}.mid'
    arranged_mt.to_midi(save_fp)


def test_drum_arranger():
    model = DrumArranger('LongshenOu/m2m_drummer', hf_ckpt=True)
    song_fp = '/data2/longshen/musecoco_data/full_song/caihong/caihong.mid'
    
    # Using preset
    preset = 'drum' # The only option
    arranged_mt = model.arrange(song_fp, use_preset=preset, merge_with_input=True) # , merge_with_input=True
    print(arranged_mt)
    save_fp = f'/data2/longshen/musecoco_data/infer_out/test_api/arranger/caihong_{preset}.mid'
    arranged_mt.to_midi(save_fp)


def procedures():
    test_band_arranger()
    test_piano_arranger()
    test_drum_arranger()


if __name__ == "__main__":
    main()

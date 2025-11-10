'''
New version
Metadata and segment-level structure for the Slakh dataset. (new)
'''

import os
import sys
dirof = os.path.dirname
sys.path.append(dirof(dirof(dirof(os.path.abspath(__file__)))))

from tqdm import tqdm
from remi_z import MultiTrack
from utils_common.utils import jpath, ls, save_json, save_yaml

def main():
    segment_dataset_1bar_norm_remi_plus()

def procedures():
    create_song_metadata()
    segment_dataset_1bar_q16_norm()
    segment_dataset_1bar_norm_withhist()
    segment_dataset_1bar_norm_remi_plus()


def create_song_metadata():
    '''
    Create song-level metadata for the Slakh dataset.
    meta = {
        'split_name': {
            'song_name': {
                'bars': #bars,
                'insts': [1, 3, 6],
            }
        }
    }
    '''
    dataset_root = '/data2/longshen/Datasets/slakh2100_flac_redux'
    save_dir = jpath(dataset_root, 'metadata')
    save_fn = 'song_metadata.yaml'
    save_fp = jpath(save_dir, save_fn)

    splits = ['test', 'validation', 'train']
    meta = {}
    for split in splits:
        print('Processing split:', split)
        split_meta = {}
        split_dir = jpath(dataset_root, 'original', split)

        song_names = ls(split_dir)
        for song_name in tqdm(song_names):
            song_dir = jpath(split_dir, song_name)
            midi_fp = jpath(song_dir, 'all_src.mid')
            
            # Read the midi file
            multitrack = MultiTrack.from_midi(midi_fp)

            insts = list(multitrack.get_unique_insts())
            insts.sort()
            insts = [str(i) for i in insts]
            insts_str = ' '.join(insts)

            n_bars = len(multitrack)

            bar0 = multitrack[0]
            ts = bar0.time_signature
            tempo = bar0.tempo

            song_meta = {
                'bars': n_bars,
                'time_signature': f'{ts}',
                'tempo': tempo,
                'insts': insts_str,
            }
            split_meta[song_name] = song_meta
        
        meta[split] = split_meta

    save_yaml(meta, save_fp)


def segment_dataset_1bar_q16_norm():
    '''
    Create segment-level dataset for the Slakh dataset.
    Meanwhile, create segment-level metadata
    '''
    dataset_root = '/data2/longshen/Datasets/slakh2100_flac_redux'
    save_dir = jpath(dataset_root, 'metadata')
    save_fn = 'segment_dataset_1bar_q16_norm.json'
    save_fp = jpath(save_dir, save_fn)
    piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])

    splits = ['test', 'validation', 'train']
    meta = {}
    for split in splits:
        print('Processing split:', split)
        split_meta = {}
        split_dir = jpath(dataset_root, 'original', split)

        song_names = ls(split_dir)
        for song_name in tqdm(song_names):
            song_dir = jpath(split_dir, song_name)
            midi_fp = jpath(song_dir, 'all_src.mid')
            
            # Read the midi file
            multitrack = MultiTrack.from_midi(midi_fp)
            multitrack.normalize_pitch()
            multitrack.quantize_to_16th()

            for bar in multitrack.bars:
                content = bar.to_remiz_seq(
                    with_tempo=True,
                    with_ts=True,
                    with_velocity=False
                )
                ts = bar.time_signature
                tempo = bar.tempo
                insts = list(bar.tracks.keys())
                insts.sort()
                insts = [str(i) for i in insts]
                insts_str = ' '.join(insts)
                bar_meta = {
                    'time_signature': f'{ts}',
                    'tempo': tempo,
                    'insts': insts_str,
                    'pitch_range': bar.get_pitch_range(),
                    'piano_pitch_range': bar.get_pitch_range(piano_ids),
                }
                split_meta[f'{song_name}-bar{bar.bar_id}'] = {
                    'content': ' '.join(content),
                    'meta': bar_meta,
                }

        meta[split] = split_meta

    save_json(meta, save_fp)


def segment_dataset_1bar_norm_remi_plus():
    '''
    Create segment-level dataset for the Slakh dataset.
    Segment length: 1bars
    Tokenization: REMI+
    With key normalized to C major or A minor.

    For tokenization efficiency comparison
    '''
    dataset_root = '/data2/longshen/Datasets/slakh2100_flac_redux'
    save_dir = jpath(dataset_root, 'metadata')
    save_fn = 'segment_dataset_1bar_norm_remiplus.json'
    save_fp = jpath(save_dir, save_fn)
    piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])

    splits = ['test', 'validation', 'train']
    meta = {}
    for split in splits:
        print('Processing split:', split)
        split_meta = {}
        split_dir = jpath(dataset_root, 'original', split)

        song_names = ls(split_dir)
        for song_name in tqdm(song_names):
            song_dir = jpath(split_dir, song_name)
            midi_fp = jpath(song_dir, 'all_src.mid')
            
            # Read the midi file
            multitrack = MultiTrack.from_midi(midi_fp)
            multitrack.normalize_pitch()
            
            entry_id = None
            for i in range(len(multitrack.bars)):
                bar = multitrack.bars[i]
                content = bar.to_remiplus_seq(
                    with_tempo=True,
                    with_ts=True,
                    with_velocity=False
                )
                content_str = ' '.join(content)
                hist_str = entry_id

                ts = bar.time_signature
                tempo = bar.tempo
                insts = list(bar.tracks.keys())
                insts.sort()
                insts = [str(i) for i in insts]
                insts_str = ' '.join(insts)
                has_drum = bar.has_drum()
                has_piano = bar.has_piano()
                bar_meta = {
                    'time_signature': f'{ts}',
                    'tempo': tempo,
                    'insts': insts_str,
                    'pitch_range': bar.get_pitch_range(),
                    'has_piano': has_piano,
                    'piano_pitch_range': bar.get_pitch_range(piano_ids),
                    'has_drum': has_drum,
                }
                entry_id = f'{song_name}-bar{bar.bar_id}'
                split_meta[entry_id] = {
                    'meta': bar_meta,
                    'content': content_str,
                    'hist': hist_str,
                }

        meta[split] = split_meta

    save_json(meta, save_fp)


def segment_dataset_1bar_norm_withhist():
    '''
    Create segment-level dataset for the Slakh dataset.
    Segment length: 2bars
    With key normalized to C major or A minor.

    For piano and band arrangement
    '''
    dataset_root = '/data2/longshen/Datasets/slakh2100_flac_redux'
    save_dir = jpath(dataset_root, 'metadata')
    save_fn = 'segment_dataset_1bar_norm_withhist.json'
    save_fp = jpath(save_dir, save_fn)
    piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])

    splits = ['test', 'validation', 'train']
    meta = {}
    for split in splits:
        print('Processing split:', split)
        split_meta = {}
        split_dir = jpath(dataset_root, 'original', split)

        song_names = ls(split_dir)
        for song_name in tqdm(song_names):
            song_dir = jpath(split_dir, song_name)
            midi_fp = jpath(song_dir, 'all_src.mid')
            
            # Read the midi file
            multitrack = MultiTrack.from_midi(midi_fp)
            multitrack.normalize_pitch()
            
            entry_id = None
            for i in range(len(multitrack.bars)):
                bar = multitrack.bars[i]
                content = bar.to_remiz_seq(
                    with_tempo=True,
                    with_ts=True,
                    with_velocity=False
                )
                content_str = ' '.join(content)
                hist_str = entry_id

                ts = bar.time_signature
                tempo = bar.tempo
                insts = list(bar.tracks.keys())
                insts.sort()
                insts = [str(i) for i in insts]
                insts_str = ' '.join(insts)
                has_drum = bar.has_drum()
                has_piano = bar.has_piano()
                bar_meta = {
                    'time_signature': f'{ts}',
                    'tempo': tempo,
                    'insts': insts_str,
                    'pitch_range': bar.get_pitch_range(),
                    'has_piano': has_piano,
                    'piano_pitch_range': bar.get_pitch_range(piano_ids),
                    'has_drum': has_drum,
                }
                entry_id = f'{song_name}-bar{bar.bar_id}'
                split_meta[entry_id] = {
                    'meta': bar_meta,
                    'content': content_str,
                    'hist': hist_str,
                }

        meta[split] = split_meta

    save_json(meta, save_fp)


def segment_dataset_4bar_norm_withhist():
    '''
    Create segment-level dataset for the Slakh dataset.
    Segment length: 4bars
    Hop bar: 1
    With key normalized to C major or A minor.

    For drum arrangement
    Deprecated
    '''
    dataset_root = '/data2/longshen/Datasets/slakh2100_flac_redux'
    save_dir = jpath(dataset_root, 'metadata')
    save_fn = 'segment_dataset_4bar_norm_withhist.json'
    save_fp = jpath(save_dir, save_fn)

    splits = ['test', 'validation', 'train']
    meta = {}
    for split in splits:
        print('Processing split:', split)
        split_meta = {}
        split_dir = jpath(dataset_root, 'original', split)

        song_names = ls(split_dir)
        seg_id = 0
        for song_name in tqdm(song_names):
            song_dir = jpath(split_dir, song_name)
            midi_fp = jpath(song_dir, 'all_src.mid')
            
            # Read the midi file
            multitrack = MultiTrack.from_midi(midi_fp)
            multitrack.normalize_pitch()

            # Insert 4 empty bars at the beginning
            multitrack.insert_empty_bars_at_front(num_bars=4)

            # Convert all bars to remiz seq
            bar_seqs = [bar.to_remiz_seq(with_tempo=True, with_ts=True, with_velocity=False) for bar in multitrack.bars]

            # Hop size: 1 bar
            for i in range(4, len(bar_seqs)-4, 1):
                bars = multitrack[i: i+4]
                content = [bar_seqs[j] for j in range(i, i+4)]
                hist_content = [bar_seqs[j] for j in range(i-4, i)]

                content = [item for sublist in content for item in sublist]
                hist_content = [item for sublist in hist_content for item in sublist]
                
                content_str = ' '.join(content)
                hist_str = ' '.join(hist_content)

                # Count bars has drum
                bars_has_drum = 0
                for bar in bars:
                    if bar.has_drum():
                        bars_has_drum += 1

                ts = bars.time_signatures[0]
                tempo = bars.tempos[0]
                insts = list(bars.get_unique_insts())
                insts.sort()
                insts = [str(i) for i in insts]
                insts_str = ' '.join(insts)
                bar_meta = {
                    'time_signature': f'{ts}',
                    'tempo': tempo,
                    'insts': insts_str,
                    'bars_has_drum': bars_has_drum,
                }
                split_meta[f'{song_name}-seg{seg_id}'] = {
                    'content': content_str,
                    'hist': hist_str,
                    'meta': bar_meta,
                }

                seg_id += 1
            
        meta[split] = split_meta

    save_json(meta, save_fp)


if __name__ == '__main__':
    main()
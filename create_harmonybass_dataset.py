import h5py
import torch
import librosa

import numpy as np

from typing import List
from tqdm.notebook import tqdm
from argparse import ArgumentParser
from scipy.interpolate import interp1d

from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor


from source.data.constants import AUDIOS_DATAPATH
from source.data.constants import (
    MIN_FREQUENCY, MAX_FREQUENCY, SAMPLING_RATE,
    DEEP_CHROMA_HOP_LENGTH, DEEP_CHROMA_FRAME_SIZE, UNIQUE_FILTERS
)

from source.data.utils import load_hooktheory_database, train_test_split_database

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def parse_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--artist', type=str, required=True, help='Which artist to create dataset upon (top100 and classical are valid options)')
    parser.add_argument('--no-split', action='store_true', help='Use this flag to not split dataset into train/valid/test sets')

    args = parser.parse_args()
    return args


def load_top100_artists_database():
    artist_names = []
    database = load_hooktheory_database(wanted_artists=None)

    for song_id, song in tqdm(database.items()):
        artist_names.append(song['metadata']['artist'])

    names, counts = np.unique(artist_names, return_counts=True)
    top100_indices = np.argsort(counts)[::-1][:100]

    top100_song_ids = set()
    top100_artists = set(names[top100_indices].tolist())

    for song_id, song in tqdm(database.items()):
        if song['metadata']['artist'] in top100_artists:
            top100_song_ids.add(song_id)

    top100_database = {song_id: song for song_id, song in database.items() if song_id in top100_song_ids}
    return top100_database


def load_classical_artists_database():
    classical_composer_names = [
        'johann-sebastian-bach', 'ludwig-van-beethoven', 'wolfgang-amadeus-mozart', 'claude-debussy',
        'antonio-vivaldi', 'edvard-grieg', 'franz-schubert', 'hans-zimmer', 'frederic-chopin',
        'john-williams', 'johannes-brahms', 'johann-pachelbel', 'pyotr-ilyich-tchaikovsky'
    ]

    classical_composer_song_ids = set()
    database = load_hooktheory_database(wanted_artists=None)

    for song_id, song in tqdm(database.items()):
        if song['metadata']['artist'] in classical_composer_names:
            classical_composer_song_ids.add(song_id)

    classical_database = {song_id: song for song_id, song in database.items() if song_id in classical_composer_song_ids}
    return classical_database

def load_artist_database(artist: str):
    if artist == 'top100':
        return load_top100_artists_database()
    elif artist == 'classical':
        return load_classical_artists_database()

    return load_hooktheory_database(wanted_artists=[artist])


def chunkify(arr: np.ndarray, window_size: int, hop_size: int, padding_value: float = 0) -> List[np.ndarray]:
    '''
    Helper function to yield chunks of data defined by window_size and hop_size.

    Arguments
    ---------
        - arr (np.ndarray): numpy array where axis=0 the sequence length information (where the data will be chunkified)
        - window_size (int): window size to be used.
        - hop_size (int): hop size to be used.
        - padding_value (float, default=0): padding value used to fill the chunk size if size is less then window size
    '''

    chunks = []
    for i in range(0, arr.shape[0], hop_size):
        chunk = arr[i:i+window_size]

        # Applying padding accordingly
        if chunk.shape[0] != window_size:
            pad = window_size - chunk.shape[0]
            chunk = np.pad(chunk, ((0, pad), (0, 0)), constant_values=padding_value)

        chunks.append(chunk)

    return chunks


def dcp_flatten(fs):
    """Flatten spectrograms for DeepChromaProcessor. Needs to be outside
       of the class in order to be picklable for multiprocessing.
    """
    return np.concatenate(fs).reshape(len(fs), -1)


def compute_log_quarter_tone_spectrogram(wave):
    if wave.dtype == np.float32:  # converting to int16
        wave = np.int16(wave / (np.max(np.abs(wave)) + 1e-6) * 32767)

    frames = FramedSignalProcessor(frame_size=DEEP_CHROMA_FRAME_SIZE, hop_size=DEEP_CHROMA_HOP_LENGTH)
    sig = SignalProcessor(num_channels=1, sample_rate=SAMPLING_RATE)
    
    stft = ShortTimeFourierTransformProcessor()
    spec = LogarithmicFilteredSpectrogramProcessor(num_bands=24, fmin=MIN_FREQUENCY,
                                                   fmax=MAX_FREQUENCY, unique_filters=UNIQUE_FILTERS)

    spec_signal = SignalProcessor(sample_rate=10)
    spec_frames = FramedSignalProcessor(frame_size=15, hop_size=1, fps=10)
    
    log_quarter_tone = spec(stft(frames(sig(wave))))
    log_quarter_tone = dcp_flatten(spec_frames(spec_signal(log_quarter_tone)))
    
    return log_quarter_tone


def compute_bass_and_harmony_labels(song, num_frames):
    beats = song['alignment']['beats']
    times = song['alignment']['times']
    times = [times[i] - times[0] for i in range(len(times))]  # making times start in 0

    beat_to_time_fn = interp1d(beats, times, kind='linear', fill_value='extrapolate')
    time_to_frame_fn = lambda time: librosa.time_to_frames(time, sr=SAMPLING_RATE, hop_length=DEEP_CHROMA_HOP_LENGTH)
    
    bass_chroma = np.zeros((12, num_frames), dtype=np.float32)
    harmony_chroma = np.zeros((12, num_frames), dtype=np.float32)
    
    for chord in song['harmony']:
        onset = time_to_frame_fn(beat_to_time_fn(chord['onset']))
        offset = time_to_frame_fn(beat_to_time_fn(chord['offset']))
    
        bass_pc = chord['bass_pitch_class']
        bass_chroma[bass_pc, onset:offset] = 1.0
        
        root_pc = chord['root_pitch_class']
        interval_semitones = chord['chord_interval_semitones']
        chord_pitch_classes = np.cumsum([root_pc] + interval_semitones) % 12
        harmony_chroma[chord_pitch_classes, onset:offset] = 1.0

    return bass_chroma.T, harmony_chroma.T


def process_harmonybass_data(database, dataset_name, split):
    datapath = f'datasets/{dataset_name}.h5'

    print(f'Processing {split} database')
    for song_id, song in tqdm(database.items()):
        with h5py.File(AUDIOS_DATAPATH, 'r') as fp:
            wave = fp[song_id][:]
            spec = compute_log_quarter_tone_spectrogram(wave)
            bass_chroma, harmony_chroma = compute_bass_and_harmony_labels(song, spec.shape[0])

        spec_chunks = chunkify(spec, window_size=128, hop_size=128, padding_value=0)
        bass_chunks = chunkify(bass_chroma, window_size=128, hop_size=128, padding_value=0)
        harmony_chunks = chunkify(harmony_chroma, window_size=128, hop_size=128, padding_value=0)

        assert len(spec_chunks) == len(bass_chunks), f'{len(spec_chunks)} != {len(bass_chunks)}'
        assert len(spec_chunks) == len(harmony_chunks), f'{len(spec_chunks)} != {len(harmony_chunks)}'

        num_chunks = len(spec_chunks)
        with h5py.File(datapath, 'a') as fp:
            for chunk_idx in range(num_chunks):
                fp.create_dataset(f'harmonybass/{split}/{song_id}__{chunk_idx}/spec', data=spec_chunks[chunk_idx], compression='gzip')
                fp.create_dataset(f'harmonybass/{split}/{song_id}__{chunk_idx}/bass_chroma', data=bass_chunks[chunk_idx], compression='gzip')
                fp.create_dataset(f'harmonybass/{split}/{song_id}__{chunk_idx}/harmony_chroma', data=harmony_chunks[chunk_idx], compression='gzip')


if __name__ == '__main__':
    args = parse_command_line_arguments()

    database = load_artist_database(args.artist)
    dataset_name = args.artist.replace('-', '_') + '_dataset'

    if args.no_split:
        process_harmonybass_data(database, dataset_name, split='whole')
    else:
        split_databases = train_test_split_database(database)
        for db, split in zip(split_databases, ['train', 'valid', 'test']):
            process_harmonybass_data(db, dataset_name, split=split)

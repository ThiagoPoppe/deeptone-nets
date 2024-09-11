import h5py
import json

from sklearn.model_selection import train_test_split
from .constants import RANDOM_SEED, AUDIOS_DATAPATH, PROCESSED_HOOKTHEORY_DATAPATH


def filter_complex_rhythm_songs(database):
    ids_to_remove = set()

    for song_id, song in database.items():
        has_meter_changes = 'METER_CHANGES' in song['metadata']['tags']
        has_swing_changes = 'SWING_CHANGES' in song['metadata']['tags']
        has_tempo_changes = 'TEMPO_CHANGES' in song['metadata']['tags']
    
        if has_meter_changes or has_swing_changes or has_tempo_changes:
            ids_to_remove.add(song_id)

    filtered_database = {k: v for k, v in database.items() if k not in ids_to_remove}
    return filtered_database


def filter_no_minmaj_scales_songs(database):
    ids_to_remove = set()
    
    for song_id, song in database.items():
        for key in song['keys']:
            if key['scale'] not in ['major', 'minor']:
                ids_to_remove.add(song_id)
                break

    filtered_database = {k: v for k, v in database.items() if k not in ids_to_remove}
    return filtered_database


def load_hooktheory_database(wanted_artists=['the-beatles'], allow_complex_rhythm=False, use_only_minmaj_scales=True):
    '''
    Helper function to load HookTheory JSON database.

    Arguments
    ---------
        - wanted_artists (list, None): artist to filter the database, if None is passed then the whole database will be returned.
        - allow_complex_rhythm (bool): if set to True the function will also return songs with METER_CHANGES, SWING_CHANGES or TEMPO_CHANGES.
        - use_only_minmaj_scales (bool): if set to False the function will also return songs that are not in the major or minor scales.

    Notes
    -----
        1. "allow_complex_rhythm" should be set to False most of the times, otherwise the alignment methods will break.
        2. "use_only_minmaj_scales" will be set to True for now, later on we can experiment with using more scales in the prediction.
        3. Will only return entries that have a downloaded audio file.
    '''

    with open(PROCESSED_HOOKTHEORY_DATAPATH, 'r') as fp:
        database = json.load(fp)

    if not allow_complex_rhythm:
        database = filter_complex_rhythm_songs(database)
    if use_only_minmaj_scales:
        database = filter_no_minmaj_scales_songs(database)

    if wanted_artists:
        wanted_artists = set(wanted_artists)
        database = {k: v for k, v in database.items() if v['metadata']['artist'] in wanted_artists}

    # Keeping songs that have audio
    with h5py.File(AUDIOS_DATAPATH, 'r') as fp:
        song_ids_with_audios = set(fp.keys())
        database = {k: v for k, v in database.items() if k in song_ids_with_audios}

    return database


def train_test_split_database(database):
    """
    Train test split will be done with respect to the song name level.
    """
    songnames = []
    for song in database.values():
        if song['metadata']['songname'] not in songnames:
            songnames.append(song['metadata']['songname'])

    train_songnames, valid_songnames = train_test_split(songnames, test_size=0.4, shuffle=True, random_state=RANDOM_SEED)
    valid_songnames, test_songnames = train_test_split(valid_songnames, test_size=0.4, shuffle=True, random_state=RANDOM_SEED)

    train_database = {song_id: song for song_id, song in database.items() if song['metadata']['songname'] in train_songnames}
    valid_database = {song_id: song for song_id, song in database.items() if song['metadata']['songname'] in valid_songnames}
    test_database = {song_id: song for song_id, song in database.items() if song['metadata']['songname'] in test_songnames}

    return train_database, valid_database, test_database

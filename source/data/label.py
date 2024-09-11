import numpy as np
from .constants import LABEL_RES
from .constants import (
    KEY_SCALE_DOMAIN, PRIMARY_DEGREES_DOMAIN, SECONDARY_DEGREES_DOMAIN,
    TRIAD_QUALITIES_DOMAIN, INVERSION_DOMAIN, ROOT_PITCH_CLASSES_DOMAIN,
    BASS_PITCH_CLASSES_DOMAIN
)


def encode_keys(keys, num_beats):
    num_beats = round(LABEL_RES * num_beats)
    key_label = np.zeros(num_beats, dtype=np.int64)

    for key in keys:
        assert key['scale'] in ['major', 'minor']

        onset = int(LABEL_RES * key['onset'])
        offset = int(LABEL_RES * key['offset'])

        tonic_pc = key['tonic_pitch_class']
        index_offset = KEY_SCALE_DOMAIN.index(key['scale'])

        key_label[onset:offset] = tonic_pc + (12 * index_offset)

    return key_label


def encode_harmony(harmony, num_beats):
    num_beats = round(LABEL_RES * num_beats)
    inversion_label = np.zeros(num_beats, dtype=np.int64)
    triad_quality_label = np.zeros(num_beats, dtype=np.int64)
    primary_degree_label = np.zeros(num_beats, dtype=np.int64)
    secondary_degree_label = np.zeros(num_beats, dtype=np.int64)
    root_pitch_class_label = np.zeros(num_beats, dtype=np.int64)
    bass_pitch_class_label = np.zeros(num_beats, dtype=np.int64)

    for chord in harmony:
        onset = int(LABEL_RES * chord['onset'])
        offset = int(LABEL_RES * chord['offset'])

        inversion_label[onset:offset] = INVERSION_DOMAIN.index(chord['inversion'])
        triad_quality_label[onset:offset] = TRIAD_QUALITIES_DOMAIN.index(chord['triad_quality'])
        primary_degree_label[onset:offset] = PRIMARY_DEGREES_DOMAIN.index(chord['primary_degree'])
        secondary_degree_label[onset:offset] = SECONDARY_DEGREES_DOMAIN.index(chord['secondary_degree'])
        root_pitch_class_label[onset:offset] = ROOT_PITCH_CLASSES_DOMAIN.index(chord['root_pitch_class'])
        bass_pitch_class_label[onset:offset] = BASS_PITCH_CLASSES_DOMAIN.index(chord['bass_pitch_class'])

    return np.stack([secondary_degree_label, primary_degree_label, triad_quality_label,
                     inversion_label, root_pitch_class_label, bass_pitch_class_label])

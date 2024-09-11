# Paths constant
# Substitute this variable with the path to your downloaded audios.
# For everything to work, you must store them in H5PY format indexed by HookTheory's song ids.
AUDIOS_DATAPATH = '/storage/datasets/thiago.poppe/functional_harmony/icassp_audios.h5'

# Substitute this variable with HookTheory's processed JSON datapath.
PROCESSED_HOOKTHEORY_DATAPATH = '/storage/datasets/thiago.poppe/functional_harmony/hooktheory_processed.json'

# Resolution
LABEL_RES = 2  # 8th note resolution
INPUT_RES = 8  # 32nd note resolution

# Step size and window size in number of quarter notes
HOP_SIZE_IN_BEATS = 32
WINDOW_SIZE_IN_BEATS = 32

# Other constants
RANDOM_SEED = 42
CHROMATIC_SCALE = 'C C# D D# E F F# G G# A A# B'.split()

# HarmonyBass Chroma Network preprocessing constants
MIN_FREQUENCY = 65
MAX_FREQUENCY = 2100
SAMPLING_RATE = 44100
UNIQUE_FILTERS = True
DEEP_CHROMA_HOP_LENGTH = 4410
DEEP_CHROMA_FRAME_SIZE = 8192

# Label related constants
KEY_SCALE_DOMAIN = ['major', 'minor']

INVERSION_DOMAIN = [0, 1, 2, 3]
TRIAD_QUALITIES_DOMAIN = ['maj', 'min', 'dim', 'aug', 'sus', '<unk>']  # <unk>: unknown token
PRIMARY_DEGREES_DOMAIN = [
    '1', '2', '3', '4', '5', '6', '7',
    '#1', '#2', '#3', '#4', '#5', '#6', '#7',
    'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'
]
SECONDARY_DEGREES_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7]
ROOT_PITCH_CLASSES_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
BASS_PITCH_CLASSES_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

LABEL_NAMES = ['key', 'secondary degree', 'primary degree',
               'triad quality', 'inversion', 'root_pc', 'bass_pc']

# We will sum 1 to harmony related labels to let last index be "no chord" label
LABEL_SIZES = [
    12 * len(KEY_SCALE_DOMAIN),
    1 + len(SECONDARY_DEGREES_DOMAIN),
    1 + len(PRIMARY_DEGREES_DOMAIN),
    1 + len(TRIAD_QUALITIES_DOMAIN),
    1 + len(INVERSION_DOMAIN),
    1 + len(ROOT_PITCH_CLASSES_DOMAIN),
    1 + len(BASS_PITCH_CLASSES_DOMAIN)
]

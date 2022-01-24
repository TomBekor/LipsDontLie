import math

# Paths
VOCABULARY_FILE = 'vocab.txt'
VIDEOS_DIR = 'npy_videos'
ANNOTATIONS_DIR = 'npy_alignments'
LANDMARKS_DIR = 'npy_landmarks'

# DataLoader
FRAME_EPSILON = 3

# General
SEQUENCE_IN_MAX_LEN = 80
SEQUENCE_OUT_MAX_LEN = 16

# Backbone hyperparameters
CONV_LAYERS_TO_FREEZE = 0

# Landmarks NN hyperparameters
INPUT_DIM = 20*2
HIDDEN_DIM = 256

# Transformer hyperparametes
TRANSFORMER_D_MODEL = 512
TRANSFORMER_N_HEADS = 4
TRANSFORMER_ENCODER_LAYERS = 2
TRANSFORMER_DECODER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 2048
TRANSFORMER_DROPOUT = 0.1

# Training
BATCH_SIZE = 16
EPOCHS = 60
VERBOSE_ITERS=100
INCREASE_DIFFICULTY_ITERS = 10
MAX_INCREASES = math.ceil(math.log2(SEQUENCE_OUT_MAX_LEN/BATCH_SIZE))
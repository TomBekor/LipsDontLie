''' A configuration file which contains constants
 and hyperparameters used in our project'''

# Paths
VOCABULARY_FILE = 'vocab.txt'
VIDEOS_DIR = 'videos'
ANNOTATIONS_DIR = 'npy_alignments'
LANDMARKS_DIR = 'npy_landmarks'

# Preprocessing
PRETRAINED_SHAPE_DETECTOR_PATH = './facial-landmarks-models/shape_predictor_68_face_landmarks.dat'

# DataLoader
SPEAKERS = list(range(2,6)) + list(range(22,29))
# SPEAKERS = list(range(2,6))
# SPEAKERS = [2]
WINDOW_SIZE = 5

# Tokenizer
SEQUENCE_IN_MAX_LEN = 80
SEQUENCE_OUT_MAX_LEN = 16

# Landmarks NN hyperparameters
INPUT_DIM = 20*2*WINDOW_SIZE
HIDDEN_DIM = 256
LMARKS_DROPOUT=0.3

# Transformer hyperparametes
TRANSFORMER_D_MODEL = 512
TRANSFORMER_N_HEADS = 4
TRANSFORMER_ENCODER_LAYERS = 2
TRANSFORMER_DECODER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 2048
TRANSFORMER_DROPOUT = 0.1

# Training
BATCH_SIZE = 32
EPOCHS = 60
VERBOSE_ITERS=100

# trained models
TRANSFORMER_PATH = 'trained_models/transformer'
LANDMARKSNN_PATH = 'trained_models/landmarks'
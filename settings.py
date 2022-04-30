import os
import pickle
import lightgbm
from lightgbm import LGBMClassifier

Emotions = {0: "neutral",
            1: "calm",
            2: "happy",
            3: "sad",
            4: "angry",
            5: "fearful",
            6: "disgust",
            7: "surprised"}

with open('dtree_pkl', 'rb') as f:
    dtree = pickle.load(f)
with open('mlp_pkl', 'rb') as f:
    MLP = pickle.load(f)
LGBM = lightgbm.Booster(model_file='lgbm.txt')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR = os.path.join(ROOT_DIR, 'output/')
RECORDING_DIR = os.path.join(OUT_DIR, 'recording')
WAVE_OUTPUT_FILE = os.path.join(RECORDING_DIR, "recorded.wav")
WAVE_PLOT_FILE = os.path.join(ROOT_DIR, 'waveplot.png')
SPECTROGRAM_FILE = os.path.join(ROOT_DIR, 'spectrogram.png')
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 44100  # Default sample rate of microphone or recording device
CHUNK_SIZE = 1024
DURATION = 5  # 3 seconds
allowed_audio = ["wav"]
options = ('LightGBM Classifier', 'Decision Tree Classifier', 'MultiLayer Perceptorn Classifier')
option = ('yes', 'no')
Models_matrix = {
    'LightGBM Classifier': LGBM,
    'Decision Tree Classifier': dtree,
    'MultiLayer Perceptorn Classifier': MLP
}

# -*- coding: utf-8 -*-
"""Fixed experiment configuration.

All values are preserved from the original single-file experiment.
"""
import os
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED_BASE = 42
N_REPEATS = 1
N_CLASSES = 2

MAX_ECG_LEN = 5000
BATCH_SIZE = 32
EPOCHS = 1
USE_EARLY_STOPPING = True
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5

D_MODEL = 64
ATTN_LEN = 64
NUM_HEADS = 4

GUMBEL_TAU_LIST = [0.3, 0.5, 0.7, 1.0, 1.5]
GATE_KL_WEIGHT_LIST = [0.0, 1e-4, 1e-3, 1e-2]
LABEL_SMOOTHING = 0.05

ECG_BASE = r"/home/dataset/mimic-iv-ecg-diagnostic"
CXR_BASE = r"/home/dataset/cxr/mimic-cxr-jpg/2.1.0"
ECG_CSV_PATH = r"/home/dataset/csv/integrated_dataset_with_ecg.csv"
CXR_CSV_PATH = r"/home/dataset/csv/integrated_cxr.csv"
TRIMODAL_CSV_PATH = r"/home/dataset/holistic_ehr_ecg_cxr_frontal.csv"
OUT_DIR = r"/home/result/TriHAR"

CAT_COLS = ["gender", "ventilation", "vasopressor", "anticoag", "beta1"]
LEVEL_NAMES = ["low", "mid", "high"]
MODALITY_NAMES = ["EHR", "ECG", "CXR"]

ECG_MODEL_NAMES = ["WaveNet", "ResNet", "LSTM"]
TAB_MODEL_NAMES = ["FTTransformer", "TabTransformer", "TPC"]
CXR_MODEL_NAMES = ["ResNet", "DenseNet", "VGG"]

NUM_WORKERS = 4

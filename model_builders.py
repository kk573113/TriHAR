# -*- coding: utf-8 -*-
from typing import Tuple

import torch.nn as nn
import torchvision.models as tv_models

from config import N_CLASSES
from heads import BaseModel, HeadBlock

# The user requested the folder name model_ecg. It is used here as the ECG package.
from model_ecg.ecg_baseline import ECG_Baseline
from model_ecg.ecg_resnet import ResNet1d
from model_ecg.wavenet import WaveNet
from model_ecg.lstm import ECG_LSTM

from model_ehr.tab_baseline import Tabular_Baseline
from model_ehr.tpc import TempPointConv
from model_ehr.tabnet import TabNet
from model_ehr.tabtransformer import TabTransformer
from model_ehr.fttransformer import FTTransformer

from model_cxr.resnet18 import CXR_ResNet18
from model_cxr.DenseNet121 import CXR_DenseNet121
from model_cxr.VGG16 import CXR_VGG16
# ============================================================
class CXR_ResNet18(BaseModel):
    def __init__(self, num_classes=N_CLASSES):
        super().__init__()
        try:
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = tv_models.resnet18(pretrained=True)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.h1 = HeadBlock(128, 64, num_classes)
        self.h2 = HeadBlock(256, 64, num_classes)
        self.h3 = HeadBlock(512, 64, num_classes)

        self.layers_groups = [
            [self.layer1],
            [self.layer2],
            [self.layer3],
            [self.layer4],
        ]

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)

        x2 = self.layer2(x1)
        f1 = x2.flatten(2)  # (B, C, H*W)
        f_seq1, f_pool1, z1 = self.h1(f1)

        x3 = self.layer3(x2)
        f2 = x3.flatten(2)
        f_seq2, f_pool2, z2 = self.h2(f2)

        x4 = self.layer4(x3)
        f3 = x4.flatten(2)
        f_seq3, f_pool3, z3 = self.h3(f3)

        return [
            z1, z2, z3,
            f_seq1, f_pool1,
            f_seq2, f_pool2,
            f_seq3, f_pool3,
        ]


# ============================================================
# TPC Config and model builders
# ============================================================
class TPCConfig:
    def __init__(self, n_layers=9, temp_kernels=None, point_sizes=None):
        if temp_kernels is None:
            temp_kernels = [2] * n_layers
        if point_sizes is None:
            point_sizes = [64] * n_layers

        self.task = "classification"
        self.n_layers = n_layers
        self.model_type = "tpc"
        self.diagnosis_size = 64
        self.main_dropout_rate = 0.3
        self.temp_dropout_rate = 0.1
        self.kernel_size = 3
        self.temp_kernels = temp_kernels
        self.point_sizes = point_sizes
        self.batchnorm = "batchnorm"
        self.no_diag = True
        self.no_mask = True
        self.no_skip_connections = False
        self.no_exp = True


def build_ecg_model(ecg_name: str):
    if ecg_name == "Baseline":
        return ECG_Baseline(num_classes=N_CLASSES)
    if ecg_name == "ResNet":
        return ResNet1d(input_channels=12, num_classes=N_CLASSES)
    if ecg_name == "WaveNet":
        return WaveNet(input_channels=12, num_classes=N_CLASSES)
    if ecg_name == "LSTM":
        return ECG_LSTM(input_channels=12, num_classes=N_CLASSES)
    raise ValueError(f"Unknown ECG model: {ecg_name}")



def build_cxr_model(cxr_name: str):
    if cxr_name == "ResNet":
        return CXR_ResNet18(num_classes=N_CLASSES)
    if cxr_name == "DenseNet":
        return CXR_DenseNet121(num_classes=N_CLASSES)
    if cxr_name == "VGG":
        return CXR_VGG16(num_classes=N_CLASSES)


def build_tab_model(tab_name: str, tab_dim: int, num_cnt: int, cat_dims: Tuple[int, ...]):
    if tab_name == "Baseline":
        return Tabular_Baseline(in_dim=tab_dim, num_classes=N_CLASSES)

    if tab_name == "TPC":
        tpc_layers = 9
        tpc_conf = TPCConfig(
            n_layers=tpc_layers,
            temp_kernels=[2] * tpc_layers,
            point_sizes=[64] * tpc_layers,
        )
        return TempPointConv(config=tpc_conf, F=tab_dim, D=0, no_flat_features=0, num_classes=N_CLASSES)

    if tab_name == "TabNet":
        return TabNet(
            input_dim=tab_dim,
            num_classes=N_CLASSES,
            cat_idxs=list(range(num_cnt, tab_dim)),
            cat_dims=list(cat_dims),
            cat_emb_dim=2,
        )

    if tab_name == "TabTransformer":
        return TabTransformer(
            categories=cat_dims,
            num_continuous=num_cnt,
            dim=32,
            depth=6,
            heads=8,
            num_classes=N_CLASSES,
        )

    if tab_name == "FTTransformer":
        return FTTransformer(
            categories=cat_dims,
            num_continuous=num_cnt,
            dim=32,
            depth=3,
            heads=8,
            num_classes=N_CLASSES,
        )

    raise ValueError(f"Unknown tabular model: {tab_name}")


# ============================================================
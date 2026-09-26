import math
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as tv_models  # <- 임포트 추가

from heads import HeadBlock, BaseModel 

N_CLASSES = 2

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
        self.layer1 = backbone.layer1  # 채널: 64
        self.layer2 = backbone.layer2  # 채널: 128
        self.layer3 = backbone.layer3  # 채널: 256
        self.layer4 = backbone.layer4  # 채널: 512

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
        x1 = self.layer1(x)  # x1 (64 채널)은 다음 레이어로 전달하는 용도로만 사용

        # Mid-low level: layer2의 결과물 활용
        x2 = self.layer2(x1)
        f1 = x2.flatten(2)   # (B, 128, H*W)
        f_seq1, f_pool1, z1 = self.h1(f1)

        # Mid-high level: layer3의 결과물 활용
        x3 = self.layer3(x2)
        f2 = x3.flatten(2)   # (B, 256, H*W)
        f_seq2, f_pool2, z2 = self.h2(f2)

        # High level: layer4의 결과물 활용
        x4 = self.layer4(x3)
        f3 = x4.flatten(2)   # (B, 512, H*W)
        f_seq3, f_pool3, z3 = self.h3(f3)

        return [
            z1, z2, z3,
            f_seq1, f_pool1,
            f_seq2, f_pool2,
            f_seq3, f_pool3,
        ]
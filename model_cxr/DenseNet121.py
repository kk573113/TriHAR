import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from heads import HeadBlock, BaseModel

N_CLASSES = 2

class CXR_DenseNet121(BaseModel):
    def __init__(self, num_classes=N_CLASSES):
        super().__init__()
        try:
            backbone = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = tv_models.densenet121(pretrained=True)

        self.features = backbone.features

        self.h1 = HeadBlock(256, 64, num_classes)
        self.h2 = HeadBlock(512, 64, num_classes)
        self.h3 = HeadBlock(1024, 64, num_classes)

        self.layers_groups = [
            [self.features.denseblock1],
            [self.features.denseblock2],
            [self.features.denseblock3, self.features.denseblock4],
        ]

    def forward(self, x):
        # ----------------------------------------------------
        # Stem: (B, 3, 224, 224) -> (B, 64, 56, 56)
        # ----------------------------------------------------
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        # ----------------------------------------------------
        # Low level: DenseBlock1 output
        # (B, 256, 56, 56)
        # ----------------------------------------------------
        x1 = self.features.denseblock1(x)

        f1 = x1.flatten(2)  # (B, 256, H*W)
        f_seq1, f_pool1, z1 = self.h1(f1)

        # (B, 256, 56, 56) -> (B, 128, 28, 28)
        x = self.features.transition1(x1)

        # ----------------------------------------------------
        # Mid level: DenseBlock2 output
        # (B, 512, 28, 28)
        # ----------------------------------------------------
        x2 = self.features.denseblock2(x)

        f2 = x2.flatten(2)  # (B, 512, H*W)
        f_seq2, f_pool2, z2 = self.h2(f2)

        # (B, 512, 28, 28) -> (B, 256, 14, 14)
        x = self.features.transition2(x2)

        # ----------------------------------------------------
        # DenseBlock3 and Transition3
        # ----------------------------------------------------
        x3 = self.features.denseblock3(x)
        # x3: (B, 1024, 14, 14)

        x3_transition = self.features.transition3(x3)
        # x3_transition: (B, 512, 7, 7)

        # ----------------------------------------------------
        # High level: DenseBlock4 output
        # (B, 1024, 7, 7)
        # ----------------------------------------------------
        x4 = self.features.denseblock4(x3_transition)

        # norm5 expects exactly 1024 channels
        x4 = self.features.norm5(x4)
        x4 = F.relu(x4, inplace=True)

        f3 = x4.flatten(2)  # (B, 1024, H*W)
        f_seq3, f_pool3, z3 = self.h3(f3)

        return [
            z1, z2, z3,
            f_seq1, f_pool1,
            f_seq2, f_pool2,
            f_seq3, f_pool3,
        ]
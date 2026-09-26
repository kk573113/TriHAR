# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as tv_models

from heads import HeadBlock, BaseModel


class CXR_VGG16(BaseModel):
    """
    VGG16 CXR encoder with low/mid/high feature outputs.

    Feature levels:
        low  : VGG block3 output, 256 channels
        mid  : VGG block4 output, 512 channels
        high : VGG block5 output, 512 channels

    Output format:
        [
            z1, z2, z3,
            f_seq1, f_pool1,
            f_seq2, f_pool2,
            f_seq3, f_pool3,
        ]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        if pretrained:
            try:
                backbone = tv_models.vgg16(
                    weights=tv_models.VGG16_Weights.IMAGENET1K_V1
                )
                print("[VGG16] Loaded ImageNet pretrained weights.")
            except Exception as exc:
                print(
                    "[VGG16] Pretrained weights unavailable. "
                    f"Using random initialization instead: {exc}"
                )
                backbone = tv_models.vgg16(weights=None)
        else:
            backbone = tv_models.vgg16(weights=None)
            print("[VGG16] Using random initialization.")

        features = backbone.features

        # VGG16 feature indices:
        #
        # block1:  0 ~ 4
        # block2:  5 ~ 9
        # block3: 10 ~ 16
        # block4: 17 ~ 23
        # block5: 24 ~ 30
        #
        self.block1 = nn.Sequential(*features[:5])
        self.block2 = nn.Sequential(*features[5:10])
        self.block3 = nn.Sequential(*features[10:17])
        self.block4 = nn.Sequential(*features[17:24])
        self.block5 = nn.Sequential(*features[24:31])

        self.h1 = HeadBlock(256, 64, num_classes)
        self.h2 = HeadBlock(512, 64, num_classes)
        self.h3 = HeadBlock(512, 64, num_classes)

        self.layers_groups = [
            [self.block1, self.block2, self.block3],
            [self.block4],
            [self.block5],
        ]

    def forward(self, x: torch.Tensor):
        # Input: (B, 3, 224, 224)

        # block1 output: (B, 64, 112, 112)
        x = self.block1(x)

        # block2 output: (B, 128, 56, 56)
        x = self.block2(x)

        # ----------------------------------------------------
        # Low-level feature
        # block3 output: (B, 256, 28, 28)
        # ----------------------------------------------------
        x3 = self.block3(x)

        f1 = x3.flatten(2).contiguous()
        f_seq1, f_pool1, z1 = self.h1(f1)

        # ----------------------------------------------------
        # Mid-level feature
        # block4 output: (B, 512, 14, 14)
        # ----------------------------------------------------
        x4 = self.block4(x3)

        f2 = x4.flatten(2).contiguous()
        f_seq2, f_pool2, z2 = self.h2(f2)

        # ----------------------------------------------------
        # High-level feature
        # block5 output: (B, 512, 7, 7)
        # ----------------------------------------------------
        x5 = self.block5(x4)

        f3 = x5.flatten(2).contiguous()
        f_seq3, f_pool3, z3 = self.h3(f3)

        return [
            z1, z2, z3,
            f_seq1, f_pool1,
            f_seq2, f_pool2,
            f_seq3, f_pool3,
        ]

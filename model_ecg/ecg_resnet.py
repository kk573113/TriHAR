import math
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from heads import HeadBlock, BaseModel  # [수정] heads.py에서 임포트

# ==========================================
# Activation Functions
# ==========================================
def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=1)
    return a * F.gelu(b)

def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    if name == 'reglu': return reglu
    if name == 'geglu': return geglu
    if name == 'sigmoid': return torch.sigmoid
    return getattr(F, name)

# ==========================================
# ResNet Block
# ==========================================
class ResNetBlock1d(nn.Module):
    def __init__(
        self,
        d: int,
        d_hidden: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        kernel_size: int = 5
    ):
        super().__init__()
        
        def make_normalization(dim):
            return nn.BatchNorm1d(dim) if normalization == 'batchnorm' else nn.GroupNorm(8, dim)

        self.activation_fn = get_activation_fn(activation)
        self.norm = make_normalization(d)
        
        scale = 2 if activation.endswith('glu') else 1
        
        self.conv0 = nn.Conv1d(d, d_hidden * scale, kernel_size=kernel_size, padding=kernel_size//2)
        self.dropout0 = nn.Dropout(hidden_dropout)
        
        self.conv1 = nn.Conv1d(d_hidden, d, kernel_size=kernel_size, padding=kernel_size//2)
        self.dropout1 = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor) -> Tensor:
        inputs = x
        z = self.norm(x)
        z = self.conv0(z)
        z = self.activation_fn(z)
        z = self.dropout0(z)
        z = self.conv1(z)
        z = self.dropout1(z)
        return inputs + z

# ==========================================
# Main ResNet1d (Multi-Head Adapted)
# ==========================================
class ResNet1d(BaseModel):  # [수정] BaseModel 상속
    def __init__(
        self,
        *,
        input_channels: int = 12,
        d: int = 64,
        d_hidden_factor: float = 2.0,
        n_layers: int = 8,
        activation: str = 'relu',
        normalization: str = 'batchnorm',
        hidden_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        num_classes: int = 2, 
        kernel_size: int = 7
    ) -> None:
        super().__init__()

        # 1. Stem
        self.first_layer = nn.Conv1d(input_channels, d, kernel_size=kernel_size, padding=kernel_size//2)

        # 2. Divide Layers into 3 Groups (Stages)
        # 예: 8 layers -> [3, 3, 2] 개씩 배분
        d_hidden = int(d * d_hidden_factor)
        
        # Helper to create a list of blocks
        def make_blocks(count):
            return nn.Sequential(*[
                ResNetBlock1d(
                    d=d, d_hidden=d_hidden, activation=activation, normalization=normalization,
                    hidden_dropout=hidden_dropout, residual_dropout=residual_dropout, kernel_size=kernel_size
                ) for _ in range(count)
            ])

        n_stage1 = n_layers // 3
        n_stage2 = n_layers // 3
        n_stage3 = n_layers - n_stage1 - n_stage2

        # Stage 1
        self.block1 = make_blocks(n_stage1)
        self.h1 = HeadBlock(d, 64, num_classes)

        # Stage 2
        self.block2 = make_blocks(n_stage2)
        self.h2 = HeadBlock(d, 64, num_classes)

        # Stage 3
        self.block3 = make_blocks(n_stage3)
        self.last_norm = nn.BatchNorm1d(d) if normalization == 'batchnorm' else nn.GroupNorm(8, d)
        self.last_act = get_activation_fn(activation)
        self.h3 = HeadBlock(d, 64, num_classes)

        # Layer Groups for Freezing
        self.layers_groups = [
            [self.first_layer, self.block1], 
            [self.block2], 
            [self.block3, self.last_norm]
        ]

    def forward(self, x: Tensor):
        # x: (Batch, 12, Length)
        
        # Stem
        x = self.first_layer(x)

        # Stage 1
        x = self.block1(x)
        # Head 1 Output
        f_seq1, f_pool1, z1 = self.h1(x)

        # Stage 2
        x = self.block2(x)
        # Head 2 Output
        f_seq2, f_pool2, z2 = self.h2(x)

        # Stage 3
        x = self.block3(x)
        x = self.last_norm(x)
        x = self.last_act(x)
        # Head 3 Output
        f_seq3, f_pool3, z3 = self.h3(x)

        # Return: [Logits(3), Logits(2), Logits(1), Feats...] (Baseline 순서와 동일하게 맞춤)
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]
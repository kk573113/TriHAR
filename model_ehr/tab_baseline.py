import torch.nn as nn
import torch.nn.functional as F
from heads import HeadBlock, BaseModel

class Tabular_Baseline(BaseModel):
    def __init__(self, in_dim, num_classes=2):  # [수정] N_CLASSES 제거 -> num_classes 인자 추가
        super().__init__()
        
        # Layer 1
        self.d1 = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.BatchNorm1d(128)
        )
        self.h1 = HeadBlock(128, 32, num_classes) # feat_dim=32

        # Layer 2
        self.d2 = nn.Sequential(
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64)
        )
        self.h2 = HeadBlock(64, 32, num_classes)

        # Layer 3
        self.d3 = nn.Sequential(
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32)
        )
        self.h3 = HeadBlock(32, 32, num_classes)

        # Layer Groups for Freezing
        self.layers_groups = [[self.d1], [self.d2], [self.d3]]

    def forward(self, x):
        # x: (Batch, in_dim)
        
        # Block 1
        x1 = F.relu(self.d1(x))
        # HeadBlock은 2D 입력이 들어오면 내부적으로 처리(unsqueeze)하도록 heads.py에 구현되어 있음
        f_seq1, f_pool1, z1 = self.h1(x1) 
        
        # Block 2
        x2 = F.relu(self.d2(x1))
        f_seq2, f_pool2, z2 = self.h2(x2)
        
        # Block 3
        x3 = F.relu(self.d3(x2))
        f_seq3, f_pool3, z3 = self.h3(x3)
        
        # Return all outputs (Logits, Features)
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]
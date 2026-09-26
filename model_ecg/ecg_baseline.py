import torch.nn as nn
from heads import HeadBlock, BaseModel

class ECG_Baseline(BaseModel):
    def __init__(self, num_classes=2):  
        super().__init__()
        
        # 1st Block
        self.c1 = nn.Sequential(
            nn.Conv1d(12, 256, 3, padding=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU()
        )
        self.h1 = HeadBlock(256, 64, num_classes)

        # 2nd Block
        self.c2 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2)
        self.h2 = HeadBlock(128, 64, num_classes)

        # 3rd Block
        self.c3 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1), 
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        self.h3 = HeadBlock(64, 64, num_classes)

        # Layer Groups for Freezing/Fine-tuning strategies
        self.layers_groups = [[self.c1], [self.c2, self.pool], [self.c3]]

    def forward(self, x):
        # x: (Batch, 12, Length)
        
        # Level 1
        x1 = self.c1(x)
        f_seq1, f_pool1, z1 = self.h1(x1)
        
        # Level 2
        x2 = self.pool(self.c2(x1))
        f_seq2, f_pool2, z2 = self.h2(x2)
        
        # Level 3
        x3 = self.c3(x2)
        f_seq3, f_pool3, z3 = self.h3(x3)
        
        # Return all intermediate outputs (Logits, Features)
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]
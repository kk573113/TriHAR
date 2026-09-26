import torch
import torch.nn as nn
from heads import HeadBlock, BaseModel

class ECG_LSTM(BaseModel):
    def __init__(
        self, 
        input_channels=12,   # ECG Lead 수
        hidden_size=128,     # LSTM 은닉층 크기
        num_classes=2,       # 출력 클래스 (3-Class LOS)
        dropout=0.2,         # 드롭아웃 비율
        bidirectional=True   # 양방향 LSTM 사용 여부
    ):
        super(ECG_LSTM, self).__init__()
        
        # LSTM의 출력 차원 계산 (양방향이면 2배)
        self.feat_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # --- Stage 1 ---
        # Input: (Batch, Length, 12) -> Output: (Batch, Length, feat_dim)
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        # HeadBlock은 (Batch, Channel, Length) 입력을 받으므로 변환 필요
        self.h1 = HeadBlock(self.feat_dim, 64, num_classes)

        # --- Stage 2 ---
        # Input: (Batch, Length, feat_dim) -> Output: (Batch, Length, feat_dim)
        self.lstm2 = nn.LSTM(
            input_size=self.feat_dim, # 이전 층의 출력을 입력으로 받음
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.h2 = HeadBlock(self.feat_dim, 64, num_classes)

        # --- Stage 3 ---
        # Input: (Batch, Length, feat_dim) -> Output: (Batch, Length, feat_dim)
        self.lstm3 = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.h3 = HeadBlock(self.feat_dim, 64, num_classes)
        
        # Layer Groups for Freezing
        self.layers_groups = [[self.lstm1], [self.lstm2], [self.lstm3]]

    def forward(self, x):
        # x input shape: (Batch, 12, Length) -> Conv1d style
        
        # 1. Permute for LSTM (Batch, Length, Features)
        x = x.permute(0, 2, 1) 
        
        # --- Stage 1 ---
        out1, _ = self.lstm1(x) 
        # LSTM output: (Batch, Length, Feat)
        # HeadBlock expects: (Batch, Feat, Length) -> Permute back
        out1_perm = out1.permute(0, 2, 1)
        f_seq1, f_pool1, z1 = self.h1(out1_perm)
        
        # --- Stage 2 ---
        out2, _ = self.lstm2(out1)
        out2_perm = out2.permute(0, 2, 1)
        f_seq2, f_pool2, z2 = self.h2(out2_perm)
        
        # --- Stage 3 ---
        out3, _ = self.lstm3(out2)
        out3_perm = out3.permute(0, 2, 1)
        f_seq3, f_pool3, z3 = self.h3(out3_perm)
        
        # Return format consistent with Baseline/ResNet
        # [Logits..., Features...]
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]
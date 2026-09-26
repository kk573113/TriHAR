import torch
import torch.nn as nn
import torch.nn.functional as F
from heads import HeadBlock, BaseModel

# =========================================================================
# WaveNet Block (Residual + Skip Connection + Gated Activation)
# =========================================================================
class WaveNetBlock(nn.Module):
    def __init__(self, residual_channels, gate_channels, skip_channels, kernel_size, dilation=1):
        super(WaveNetBlock, self).__init__()
        
        # Dilated Convolution
        self.dilated = nn.Conv1d(
            residual_channels, 
            gate_channels * 2, # For Gated Activation (Filter + Gate)
            kernel_size, 
            dilation=dilation, 
            padding=(dilation * (kernel_size - 1)) // 2
        )
        
        # 1x1 Convolution for Residual connection
        self.conv_res = nn.Conv1d(gate_channels, residual_channels, 1)
        
        # 1x1 Convolution for Skip connection
        self.conv_skip = nn.Conv1d(gate_channels, skip_channels, 1)

    def forward(self, x):
        # x: (B, C, L)
        
        # 1. Dilated Conv
        out = self.dilated(x)
        
        # 2. Gated Activation Unit
        tan_out, sig_out = out.chunk(2, dim=1)
        out = torch.tanh(tan_out) * torch.sigmoid(sig_out)
        
        # 3. Skip Connection Output
        skip = self.conv_skip(out)
        
        # 4. Residual Output (Add to input)
        res = self.conv_res(out)
        
        return (x + res), skip

# =========================================================================
# Main WaveNet Class (Multi-Head Adapted)
# =========================================================================
class WaveNet(BaseModel):
    def __init__(
        self, 
        input_channels=12,      # ECG Lead 수
        residual_channels=32,   # Residual Block 내부 채널 수
        gate_channels=32,       # Gated Activation 채널 수
        skip_channels=64,       # Skip Connection 채널 수
        num_classes=2,          # 3-Class Output
        num_blocks=3,           # Dilation Cycle 반복 횟수 (3 Cycle)
        num_layers=4,           # Cycle당 Layer 수 (예: 4라면 1, 2, 4, 8)
        kernel_size=3
    ):
        super(WaveNet, self).__init__()
        
        self.start_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        # Helper to create a sequence of WaveNet blocks
        def make_stage_blocks(n_cycle_start, n_cycles):
            blocks = nn.ModuleList()
            for b in range(n_cycle_start, n_cycle_start + n_cycles):
                for i in range(num_layers):
                    dilation = 2 ** i
                    blocks.append(
                        WaveNetBlock(residual_channels, gate_channels, skip_channels, kernel_size, dilation)
                    )
            return blocks

        # Divide blocks into 3 Stages
        # num_blocks(Cycle)를 3등분
        # 예: num_blocks=3이면 각 Stage당 1 Cycle
        n_stage1 = num_blocks // 3
        n_stage2 = num_blocks // 3
        n_stage3 = num_blocks - n_stage1 - n_stage2
        
        self.stage1_blocks = make_stage_blocks(0, n_stage1)
        self.stage2_blocks = make_stage_blocks(n_stage1, n_stage2)
        self.stage3_blocks = make_stage_blocks(n_stage1 + n_stage2, n_stage3)

        # Heads for each stage
        # WaveNet의 Skip connection 합은 (B, skip_channels, L) 형태이므로
        # HeadBlock은 in_dim=skip_channels를 받음
        self.h1 = HeadBlock(skip_channels, 64, num_classes)
        self.h2 = HeadBlock(skip_channels, 64, num_classes)
        self.h3 = HeadBlock(skip_channels, 64, num_classes)
        
        # Layer Groups for Freezing
        self.layers_groups = [
            [self.start_conv, self.stage1_blocks], 
            [self.stage2_blocks], 
            [self.stage3_blocks]
        ]

    def process_stage(self, x, blocks, prev_skip_sum=None):
        """
        한 Stage의 Block들을 통과하며 Residual과 Skip connection을 처리
        """
        current_skips = []
        for block in blocks:
            x, skip = block(x)
            current_skips.append(skip)
        
        # 현재 Stage의 Skip 합
        stage_skip_sum = sum(current_skips)
        
        # 이전 Stage까지의 Skip 합과 누적 (WaveNet은 전체 Skip 합을 사용하므로)
        if prev_skip_sum is not None:
            total_skip_sum = prev_skip_sum + stage_skip_sum
        else:
            total_skip_sum = stage_skip_sum
            
        return x, total_skip_sum

    def forward(self, x):
        # x: (Batch, 12, Length)
        x = self.start_conv(x)
        
        # --- Stage 1 ---
        x, skip_sum1 = self.process_stage(x, self.stage1_blocks, prev_skip_sum=None)
        # Skip connection의 합을 Head에 통과시켜 예측
        # Skip Sum: (B, skip_channels, L)
        f_seq1, f_pool1, z1 = self.h1(skip_sum1)
        
        # --- Stage 2 ---
        x, skip_sum2 = self.process_stage(x, self.stage2_blocks, prev_skip_sum=skip_sum1)
        f_seq2, f_pool2, z2 = self.h2(skip_sum2)
        
        # --- Stage 3 ---
        x, skip_sum3 = self.process_stage(x, self.stage3_blocks, prev_skip_sum=skip_sum2)
        f_seq3, f_pool3, z3 = self.h3(skip_sum3)
        
        # Return format consistent with other models
        # [Logits..., Features...]
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadBlock(nn.Module):
    def __init__(self, in_dim, feat_dim=64, num_classes=3):
        super().__init__()
        self.conv_feat = nn.Conv1d(in_dim, feat_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.fc_head = nn.Linear(feat_dim, num_classes)  # logits

    def forward(self, x):
        if x.dim() == 3:
            # (Batch, Channel, Length) -> (Batch, Feat, Length)
            feat_seq = F.relu(self.bn(self.conv_feat(x)))
            # Global Average Pooling: (Batch, Feat)
            feat_pool = feat_seq.mean(dim=-1)
        else:
            # (Batch, Channel) -> (Batch, Channel, 1) -> ...
            x = x.unsqueeze(-1)
            feat_seq = F.relu(self.bn(self.conv_feat(x)))
            feat_pool = feat_seq.squeeze(-1)

        logits = self.fc_head(feat_pool)  # (Batch, Classes)
        return feat_seq, feat_pool, logits

class BaseModel(nn.Module):
    """
    모델의 일부 레이어를 동결(Freeze)하는 유틸리티 메서드를 포함한 베이스 클래스
    """
    def _freeze_modules(self, modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    def freeze_by_index(self, idx):
        if hasattr(self, 'layers_groups'):
            for i in range(idx + 1):
                if i < len(self.layers_groups):
                    self._freeze_modules(self.layers_groups[i])
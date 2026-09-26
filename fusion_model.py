# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ATTN_LEN, D_MODEL, N_CLASSES, NUM_HEADS

# ============================================================
def unpack_encoder_outputs(outs: List[torch.Tensor]):
    """
    Encoder output format:
    [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]
    """
    logits = [outs[0], outs[1], outs[2]]
    seqs = [outs[3], outs[5], outs[7]]
    pools = [outs[4], outs[6], outs[8]]
    return logits, seqs, pools


class LevelHardGate(nn.Module):
    """
    Selects one low/mid/high level per modality using straight-through Gumbel-Softmax.
    The selected sequence representation is projected to D_MODEL and resampled to ATTN_LEN.
    """
    def __init__(self, d_model=D_MODEL, attn_len=ATTN_LEN, tau=1.0):
        super().__init__()
        self.d_model = d_model
        self.attn_len = attn_len
        self.tau = tau

        self.level_scorers = nn.ModuleList([nn.LazyLinear(1) for _ in range(3)])
        self.seq_projs = nn.ModuleList([nn.LazyLinear(d_model) for _ in range(3)])
        self.pool_projs = nn.ModuleList([nn.LazyLinear(d_model) for _ in range(3)])

    def _standardize_seq(self, x):
        # x must be (B, L, D)
        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.attn_len)
        x = x.transpose(1, 2)
        return x

    def _standardize_pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # If sequence-like, average over length dimension after converting to (B, L, D)
            x = self._standardize_seq(x).mean(dim=1)
        elif x.dim() > 3:
            x = x.flatten(1)
        return x

    def forward(self, logits_list, seq_list, pool_list):
        # Gate logits from pooled features
        score_list = []
        pooled_proj_list = []
        seq_proj_list = []

        for i in range(3):
            p = self._standardize_pool(pool_list[i])
            score_list.append(self.level_scorers[i](p))
            pooled_proj_list.append(self.pool_projs[i](p))

            s = self._standardize_seq(seq_list[i])
            seq_proj_list.append(self.seq_projs[i](s))

        level_logits = torch.cat(score_list, dim=1)  # (B, 3)

        if self.training:
            hard_gate = F.gumbel_softmax(level_logits, tau=self.tau, hard=True, dim=-1)
        else:
            hard_gate = F.one_hot(torch.argmax(level_logits, dim=-1), num_classes=3).float()

        # Prediction from selected head logits
        z_stack = torch.stack(logits_list, dim=1)  # (B, 3, C)
        selected_logits = torch.sum(hard_gate.unsqueeze(-1) * z_stack, dim=1)
        selected_prob = F.softmax(selected_logits, dim=-1)

        # Selected sequence and pooled representation
        seq_stack = torch.stack(seq_proj_list, dim=1)  # (B, 3, L, D)
        selected_seq = torch.sum(hard_gate.view(hard_gate.size(0), 3, 1, 1) * seq_stack, dim=1)

        pool_stack = torch.stack(pooled_proj_list, dim=1)  # (B, 3, D)
        selected_pool = torch.sum(hard_gate.unsqueeze(-1) * pool_stack, dim=1)

        return {
            "selected_logits": selected_logits,
            "selected_prob": selected_prob,
            "selected_seq": selected_seq,
            "selected_pool": selected_pool,
            "level_gate": hard_gate,
            "level_logits": level_logits,
        }


class SoftMoEGate(nn.Module):
    """Soft gate over a set of expert representations."""
    def __init__(self, n_experts: int, d_model=D_MODEL, hidden=64):
        super().__init__()
        self.n_experts = n_experts
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, expert_reprs: torch.Tensor) -> torch.Tensor:
        # expert_reprs: (B, E, D)
        logits = self.net(expert_reprs).squeeze(-1)  # (B, E)
        return F.softmax(logits, dim=-1)


class AttentionPooling(nn.Module):
    """Learnable attention pooling over a sequence (B, L, D) -> (B, D)."""
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"AttentionPooling expects (B,L,D), got {tuple(x.shape)}")
        scores = self.score(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return pooled, weights


class EHRAnchorAttentionMoE(nn.Module):
    """
    Module 2.
    - EHR <- ECG
    - ECG <- EHR
    - EHR <- CXR
    - CXR <- EHR
    - EHR_enh = LayerNorm(EHR + MLP([EHR<-ECG; EHR<-CXR]))
    - Attention pooling instead of MeanPool
    - Soft MoE over expert logits
    """
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS):
        super().__init__()
        self.attn_ehr_from_ecg = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_ecg_from_ehr = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_ehr_from_cxr = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_cxr_from_ehr = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.ehr_enh_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
        )
        self.ehr_enh_norm = nn.LayerNorm(d_model)

        self.pool_ehr_enh = AttentionPooling(d_model)
        self.pool_ecg_ehr = AttentionPooling(d_model)
        self.pool_cxr_ehr = AttentionPooling(d_model)

        self.clf_ehr_enh = nn.Linear(d_model, N_CLASSES)
        self.clf_ecg_ehr = nn.Linear(d_model, N_CLASSES)
        self.clf_cxr_ehr = nn.Linear(d_model, N_CLASSES)

        self.attn_expert_gate = SoftMoEGate(n_experts=3, d_model=d_model)

    def forward(self, ehr_seq, ecg_seq, cxr_seq):
        a_ehr_ecg, _ = self.attn_ehr_from_ecg(
            query=ehr_seq, key=ecg_seq, value=ecg_seq
        )
        a_ecg_ehr, _ = self.attn_ecg_from_ehr(
            query=ecg_seq, key=ehr_seq, value=ehr_seq
        )
        a_ehr_cxr, _ = self.attn_ehr_from_cxr(
            query=ehr_seq, key=cxr_seq, value=cxr_seq
        )
        a_cxr_ehr, _ = self.attn_cxr_from_ehr(
            query=cxr_seq, key=ehr_seq, value=ehr_seq
        )

        # Preserve the original EHR after cross-modal enhancement.
        ehr_fused = self.ehr_enh_fuse(
            torch.cat([a_ehr_ecg, a_ehr_cxr], dim=-1)
        )
        ehr_enh_seq = self.ehr_enh_norm(ehr_seq + ehr_fused)

        # Learnable sequence aggregation; no arithmetic MeanPool.
        r_ehr_enh, w_ehr_enh = self.pool_ehr_enh(ehr_enh_seq)
        r_ecg_ehr, w_ecg_ehr = self.pool_ecg_ehr(a_ecg_ehr)
        r_cxr_ehr, w_cxr_ehr = self.pool_cxr_ehr(a_cxr_ehr)

        z_ehr_enh = self.clf_ehr_enh(r_ehr_enh)
        z_ecg_ehr = self.clf_ecg_ehr(r_ecg_ehr)
        z_cxr_ehr = self.clf_cxr_ehr(r_cxr_ehr)

        expert_logits = torch.stack(
            [z_ehr_enh, z_ecg_ehr, z_cxr_ehr], dim=1
        )
        expert_reprs = torch.stack(
            [r_ehr_enh, r_ecg_ehr, r_cxr_ehr], dim=1
        )
        beta = self.attn_expert_gate(expert_reprs)

        z_m2 = torch.sum(beta.unsqueeze(-1) * expert_logits, dim=1)
        p_m2 = F.softmax(z_m2, dim=-1)

        return {
            "z_m2": z_m2,
            "p_m2": p_m2,
            "attn_expert_logits": expert_logits,
            "attn_expert_probs": F.softmax(expert_logits, dim=-1),
            "attn_expert_gate": beta,
            "pool_weights": {
                "ehr_enh": w_ehr_enh,
                "ecg_from_ehr": w_ecg_ehr,
                "cxr_from_ehr": w_cxr_ehr,
            },
            "reprs": {
                "ehr_enh": r_ehr_enh,
                "ecg_from_ehr": r_ecg_ehr,
                "cxr_from_ehr": r_cxr_ehr,
            },
        }


class TrimodalLeMoFMoE(nn.Module):
    def __init__(
        self,
        ecg_encoder,
        ehr_encoder,
        cxr_encoder,
        d_model=D_MODEL,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ehr_encoder = ehr_encoder
        self.cxr_encoder = cxr_encoder
        self.gumbel_tau = float(gumbel_tau)

        self.ecg_level_gate = LevelHardGate(d_model=d_model, tau=self.gumbel_tau)
        self.ehr_level_gate = LevelHardGate(d_model=d_model, tau=self.gumbel_tau)
        self.cxr_level_gate = LevelHardGate(d_model=d_model, tau=self.gumbel_tau)

        # Module 1: LayerNorm is used only for modality-gate inputs.
        self.modality_gate_norm = nn.LayerNorm(d_model)
        self.modality_gate = SoftMoEGate(n_experts=3, d_model=d_model)

        self.module2 = EHRAnchorAttentionMoE(d_model=d_model)

        # Learnable fusion of Module 1 and Module 2 logits.
        self.final_fusion = nn.Sequential(
            nn.Linear(N_CLASSES * 2, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, N_CLASSES),
        )

    def forward(self, x_ecg, x_cxr, x_ehr):
        ecg_outs = self.ecg_encoder(x_ecg)
        ehr_outs = self.ehr_encoder(x_ehr)
        cxr_outs = self.cxr_encoder(x_cxr)

        ecg_logits, ecg_seqs, ecg_pools = unpack_encoder_outputs(ecg_outs)
        ehr_logits, ehr_seqs, ehr_pools = unpack_encoder_outputs(ehr_outs)
        cxr_logits, cxr_seqs, cxr_pools = unpack_encoder_outputs(cxr_outs)

        ecg_sel = self.ecg_level_gate(ecg_logits, ecg_seqs, ecg_pools)
        ehr_sel = self.ehr_level_gate(ehr_logits, ehr_seqs, ehr_pools)
        cxr_sel = self.cxr_level_gate(cxr_logits, cxr_seqs, cxr_pools)

        # Module 1: gate-normalized representations, weighted logit fusion.
        modality_reprs = torch.stack([
            ehr_sel["selected_pool"],
            ecg_sel["selected_pool"],
            cxr_sel["selected_pool"],
        ], dim=1)
        modality_logits = torch.stack([
            ehr_sel["selected_logits"],
            ecg_sel["selected_logits"],
            cxr_sel["selected_logits"],
        ], dim=1)

        alpha = self.modality_gate(
            self.modality_gate_norm(modality_reprs)
        )
        z_m1 = torch.sum(alpha.unsqueeze(-1) * modality_logits, dim=1)
        p_m1 = F.softmax(z_m1, dim=-1)

        # Module 2: residual enhanced EHR, attention pooling, logit MoE.
        m2 = self.module2(
            ehr_seq=ehr_sel["selected_seq"],
            ecg_seq=ecg_sel["selected_seq"],
            cxr_seq=cxr_sel["selected_seq"],
        )
        z_m2 = m2["z_m2"]
        p_m2 = m2["p_m2"]

        # Final output is a probability, while CE is computed from z_final.
        z_final = self.final_fusion(torch.cat([z_m1, z_m2], dim=-1))
        p_final = F.softmax(z_final, dim=-1)

        return {
            "z_final": z_final,
            "p_final": p_final,
            "z_m1": z_m1,
            "p_m1": p_m1,
            "z_m2": z_m2,
            "p_m2": p_m2,
            "selected": {
                "EHR": ehr_sel,
                "ECG": ecg_sel,
                "CXR": cxr_sel,
            },
            "modality_gate": alpha,
            "attn_expert_gate": m2["attn_expert_gate"],
        }


# ============================================================
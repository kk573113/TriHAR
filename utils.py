# -*- coding: utf-8 -*-
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

from config import N_CLASSES, LEVEL_NAMES, MODALITY_NAMES

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def cm_to_cols(cm, n_cls=N_CLASSES, prefix="CM") -> Dict[str, int]:
    cm = np.asarray(cm, dtype=int)
    out = {}
    for i in range(n_cls):
        for j in range(n_cls):
            out[f"{prefix}_{i}{j}"] = int(cm[i, j])
    return out


def get_clf_eval_mc(name: str, y_true, proba, elapsed_sec: float = 0.0) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )

    if N_CLASSES == 2:
        try:
            auc = roc_auc_score(y_true, proba[:, 1])
        except ValueError:
            auc = np.nan
    else:
        y_onehot = np.eye(N_CLASSES)[y_true]
        auc = roc_auc_score(y_onehot, proba, average="macro", multi_class="ovr")

    cm = confusion_matrix(y_true, pred, labels=list(range(N_CLASSES)))
    row = {
        "Model": name,
        "Accuracy": float(acc),
        "Balanced_Acc": float(bacc),
        "Macro_Precision": float(prec),
        "Macro_Recall": float(rec),
        "Macro_F1": float(f1),
        "Macro_ROC_AUC(ovr)": float(auc),
        "Elapsed_sec": float(elapsed_sec),
    }
    row.update(cm_to_cols(cm, n_cls=N_CLASSES, prefix="CM"))
    return row


def smooth_ce_from_probs(probs: torch.Tensor, y: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Cross entropy for already-softmaxed probability vectors."""
    probs = probs.clamp_min(1e-8)
    log_probs = torch.log(probs)

    if smoothing <= 0.0:
        return F.nll_loss(log_probs, y)

    with torch.no_grad():
        true_dist = torch.zeros_like(probs)
        true_dist.fill_(smoothing / (N_CLASSES - 1))
        true_dist.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)

    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def kl_to_uniform(weights: torch.Tensor) -> torch.Tensor:
    """KL(weight distribution || uniform distribution). Lower means less collapsed."""
    eps = 1e-8
    n = weights.size(-1)
    uniform = torch.full_like(weights, 1.0 / n)
    return torch.sum(weights.clamp_min(eps) * (torch.log(weights.clamp_min(eps)) - torch.log(uniform)), dim=-1).mean()


def fmt_float_for_path(value: float) -> str:
    """Convert a float to a compact filesystem-safe string."""
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        text = f"{value:.0e}"
    else:
        text = f"{value:g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def make_experiment_name(gate_kl_weight: float, gumbel_tau: float) -> str:
    return (
        f"gateKL_{fmt_float_for_path(gate_kl_weight)}"
        f"__tau_{fmt_float_for_path(gumbel_tau)}"
    )


def summarize_gates(
    gates: Dict[str, np.ndarray],
    repeat: int,
    model_name: str,
    gate_kl_weight: float,
    gumbel_tau: float,
) -> pd.DataFrame:
    rows = []

    for key, labels in [
        ("ehr_level", LEVEL_NAMES),
        ("ecg_level", LEVEL_NAMES),
        ("cxr_level", LEVEL_NAMES),
        ("modality_gate", MODALITY_NAMES),
        ("attn_expert_gate", ["EHR_enh", "ECG_from_EHR", "CXR_from_EHR"]),
    ]:
        arr = gates[key]
        mean_w = arr.mean(axis=0)
        for i, label in enumerate(labels):
            rows.append({
                "Repeat": repeat,
                "Model": model_name,
                "Gate_KL_Weight": float(gate_kl_weight),
                "Gumbel_Tau": float(gumbel_tau),
                "Gate": key,
                "Expert": label,
                "Mean_Weight": float(mean_w[i]),
            })

    return pd.DataFrame(rows)

def save_sample_level_outputs(
    save_path: str,
    y_true: np.ndarray,
    probas: Dict[str, np.ndarray],
    gates: Dict[str, np.ndarray],
    gate_kl_weight: float,
    gumbel_tau: float,
):
    rows = []

    p_final = probas["Final"]
    p_m1 = probas["Module1"]
    p_m2 = probas["Module2"]

    ehr_level = gates["ehr_level"]          # (N, 3)
    ecg_level = gates["ecg_level"]          # (N, 3)
    cxr_level = gates["cxr_level"]          # (N, 3)
    modality_gate = gates["modality_gate"]  # (N, 3)
    attn_gate = gates["attn_expert_gate"]   # (N, 3)

    for i in range(len(y_true)):
        row = {
            "sample_index": i,
            "gate_kl_weight": float(gate_kl_weight),
            "gumbel_tau": float(gumbel_tau),
            "y_true": int(y_true[i]),

            "final_pred": int(np.argmax(p_final[i])),
            "final_prob_0": float(p_final[i, 0]),
            "final_prob_1": float(p_final[i, 1]),

            "module1_pred": int(np.argmax(p_m1[i])),
            "module1_prob_0": float(p_m1[i, 0]),
            "module1_prob_1": float(p_m1[i, 1]),

            "module2_pred": int(np.argmax(p_m2[i])),
            "module2_prob_0": float(p_m2[i, 0]),
            "module2_prob_1": float(p_m2[i, 1]),

            # selected level index
            "ehr_selected_level": LEVEL_NAMES[int(np.argmax(ehr_level[i]))],
            "ecg_selected_level": LEVEL_NAMES[int(np.argmax(ecg_level[i]))],
            "cxr_selected_level": LEVEL_NAMES[int(np.argmax(cxr_level[i]))],

            # EHR level hard gate
            "ehr_level_low": float(ehr_level[i, 0]),
            "ehr_level_mid": float(ehr_level[i, 1]),
            "ehr_level_high": float(ehr_level[i, 2]),

            # ECG level hard gate
            "ecg_level_low": float(ecg_level[i, 0]),
            "ecg_level_mid": float(ecg_level[i, 1]),
            "ecg_level_high": float(ecg_level[i, 2]),

            # CXR level hard gate
            "cxr_level_low": float(cxr_level[i, 0]),
            "cxr_level_mid": float(cxr_level[i, 1]),
            "cxr_level_high": float(cxr_level[i, 2]),

            # Module 1 modality soft gate
            "m1_gate_EHR": float(modality_gate[i, 0]),
            "m1_gate_ECG": float(modality_gate[i, 1]),
            "m1_gate_CXR": float(modality_gate[i, 2]),

            # Module 2 attention-expert soft gate
            "m2_gate_EHR_enh": float(attn_gate[i, 0]),
            "m2_gate_ECG_from_EHR": float(attn_gate[i, 1]),
            "m2_gate_CXR_from_EHR": float(attn_gate[i, 2]),
        }

        rows.append(row)

    pd.DataFrame(rows).to_csv(save_path, index=False)



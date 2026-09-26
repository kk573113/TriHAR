# -*- coding: utf-8 -*-
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DEVICE, EPOCHS, LABEL_SMOOTHING, LR, PATIENCE, USE_EARLY_STOPPING, WEIGHT_DECAY
from utils import get_clf_eval_mc, kl_to_uniform

def train_one_model(
    model,
    train_loader,
    val_loader,
    class_weights=None,
    epochs=EPOCHS,
    patience=PATIENCE,
    use_early_stopping=USE_EARLY_STOPPING,
    gate_kl_weight: float = 0.0,
    gumbel_tau: float = 1.0,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)

    best_auc = -float("inf")
    best_state = None
    best_epoch = 0
    counter = 0
    history = []

    print(
        f"Training configuration | epochs={epochs}, "
        f"early_stopping={use_early_stopping}, patience={patience}, "
        f"criterion=validation Final AUC, "
        f"gate_kl_weight={gate_kl_weight}, gumbel_tau={gumbel_tau}"
    )

    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        sums = {
            "total": 0.0,
            "final": 0.0,
            "m1": 0.0,
            "m2": 0.0,
            "gate": 0.0,
        }
        n_tr = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {ep:03d}/{epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for x_ecg, x_cxr, x_tab, y in train_bar:
            x_ecg = x_ecg.to(DEVICE, non_blocking=True)
            x_cxr = x_cxr.to(DEVICE, non_blocking=True)
            x_tab = x_tab.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            out = model(x_ecg, x_cxr, x_tab)

            loss_final = F.cross_entropy(
                out["z_final"], y,
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )
            loss_m1 = F.cross_entropy(
                out["z_m1"], y,
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )
            loss_m2 = F.cross_entropy(
                out["z_m2"], y,
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )
            gate_loss = (
                kl_to_uniform(out["modality_gate"])
                + kl_to_uniform(out["attn_expert_gate"])
            )

            loss = (
                loss_final
                + 0.3 * loss_m1
                + 0.3 * loss_m2
                + gate_kl_weight * gate_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            bs = y.size(0)
            n_tr += bs
            sums["total"] += loss.item() * bs
            sums["final"] += loss_final.item() * bs
            sums["m1"] += loss_m1.item() * bs
            sums["m2"] += loss_m2.item() * bs
            sums["gate"] += gate_loss.item() * bs

            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "Lf": f"{loss_final.item():.3f}",
                "L1": f"{loss_m1.item():.3f}",
                "L2": f"{loss_m2.item():.3f}",
                "grad": f"{float(grad_norm):.2f}",
            })

        val_metrics, val_loss, val_gates = evaluate(
            model, val_loader, return_loss=True, class_weights=class_weights
        )
        final_m = val_metrics["Final"]
        m1_m = val_metrics["Module1"]
        m2_m = val_metrics["Module2"]
        val_auc = final_m["Macro_ROC_AUC(ovr)"]

        row = {
            "epoch": ep,
            "gate_kl_weight": float(gate_kl_weight),
            "gumbel_tau": float(gumbel_tau),
            "train_loss": sums["total"] / max(n_tr, 1),
            "train_final_loss": sums["final"] / max(n_tr, 1),
            "train_m1_loss": sums["m1"] / max(n_tr, 1),
            "train_m2_loss": sums["m2"] / max(n_tr, 1),
            "train_gate_loss": sums["gate"] / max(n_tr, 1),
            "val_loss": val_loss,
            "val_final_acc": final_m["Accuracy"],
            "val_final_balanced_acc": final_m["Balanced_Acc"],
            "val_final_f1": final_m["Macro_F1"],
            "val_final_auc": val_auc,
            "val_m1_acc": m1_m["Accuracy"],
            "val_m1_balanced_acc": m1_m["Balanced_Acc"],
            "val_m1_f1": m1_m["Macro_F1"],
            "val_m1_auc": m1_m["Macro_ROC_AUC(ovr)"],
            "val_m2_acc": m2_m["Accuracy"],
            "val_m2_balanced_acc": m2_m["Balanced_Acc"],
            "val_m2_f1": m2_m["Macro_F1"],
            "val_m2_auc": m2_m["Macro_ROC_AUC(ovr)"],
            "val_ehr_low": float(val_gates["ehr_level"].mean(axis=0)[0]),
            "val_ehr_mid": float(val_gates["ehr_level"].mean(axis=0)[1]),
            "val_ehr_high": float(val_gates["ehr_level"].mean(axis=0)[2]),
            "val_ecg_low": float(val_gates["ecg_level"].mean(axis=0)[0]),
            "val_ecg_mid": float(val_gates["ecg_level"].mean(axis=0)[1]),
            "val_ecg_high": float(val_gates["ecg_level"].mean(axis=0)[2]),
            "val_cxr_low": float(val_gates["cxr_level"].mean(axis=0)[0]),
            "val_cxr_mid": float(val_gates["cxr_level"].mean(axis=0)[1]),
            "val_cxr_high": float(val_gates["cxr_level"].mean(axis=0)[2]),
            "val_m1_gate_ehr": float(val_gates["modality_gate"].mean(axis=0)[0]),
            "val_m1_gate_ecg": float(val_gates["modality_gate"].mean(axis=0)[1]),
            "val_m1_gate_cxr": float(val_gates["modality_gate"].mean(axis=0)[2]),
            "val_m2_gate_ehr_enh": float(val_gates["attn_expert_gate"].mean(axis=0)[0]),
            "val_m2_gate_ecg_ehr": float(val_gates["attn_expert_gate"].mean(axis=0)[1]),
            "val_m2_gate_cxr_ehr": float(val_gates["attn_expert_gate"].mean(axis=0)[2]),
            "epoch_sec": time.time() - epoch_start,
        }
        history.append(row)

        print(
            f"\n[Epoch {ep:03d}/{epochs}] "
            f"GateKL={gate_kl_weight:g} | Tau={gumbel_tau:g} | "
            f"train={row['train_loss']:.4f} "
            f"(Final={row['train_final_loss']:.4f}, "
            f"M1={row['train_m1_loss']:.4f}, M2={row['train_m2_loss']:.4f}) | "
            f"val_loss={val_loss:.4f} | time={row['epoch_sec']:.1f}s"
        )
        print(
            f"  Final | Acc={row['val_final_acc']:.4f} | "
            f"BAcc={row['val_final_balanced_acc']:.4f} | "
            f"F1={row['val_final_f1']:.4f} | AUC={val_auc:.4f}"
        )
        print(
            f"  M1    | Acc={row['val_m1_acc']:.4f} | "
            f"BAcc={row['val_m1_balanced_acc']:.4f} | "
            f"F1={row['val_m1_f1']:.4f} | AUC={row['val_m1_auc']:.4f}"
        )
        print(
            f"  M2    | Acc={row['val_m2_acc']:.4f} | "
            f"BAcc={row['val_m2_balanced_acc']:.4f} | "
            f"F1={row['val_m2_f1']:.4f} | AUC={row['val_m2_auc']:.4f}"
        )
        print(
            "  Level | "
            f"EHR={row['val_ehr_low']:.2f}/{row['val_ehr_mid']:.2f}/{row['val_ehr_high']:.2f}, "
            f"ECG={row['val_ecg_low']:.2f}/{row['val_ecg_mid']:.2f}/{row['val_ecg_high']:.2f}, "
            f"CXR={row['val_cxr_low']:.2f}/{row['val_cxr_mid']:.2f}/{row['val_cxr_high']:.2f}"
        )
        print(
            "  Gates | "
            f"M1(EHR/ECG/CXR)={row['val_m1_gate_ehr']:.2f}/"
            f"{row['val_m1_gate_ecg']:.2f}/{row['val_m1_gate_cxr']:.2f}, "
            f"M2(EHRenh/ECG<-EHR/CXR<-EHR)={row['val_m2_gate_ehr_enh']:.2f}/"
            f"{row['val_m2_gate_ecg_ehr']:.2f}/{row['val_m2_gate_cxr_ehr']:.2f}"
        )

        improved = not np.isnan(val_auc) and val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            counter = 0
            print(f"  -> Best validation AUC updated: {best_auc:.5f}")
        else:
            counter += 1
            print(
                f"  -> Validation AUC did not improve: {counter}/{patience} "
                f"(best={best_auc:.5f} at epoch {best_epoch})"
            )
            if use_early_stopping and counter >= patience:
                print(
                    f"Early stopping at epoch {ep}: validation AUC did not "
                    f"improve for {patience} consecutive epochs."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch}, AUC={best_auc:.5f}")

    return model, pd.DataFrame(history), best_epoch, best_auc


@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    ys = []
    p_final_list, p_m1_list, p_m2_list = [], [], []

    gate_records = {
        "ehr_level": [],
        "ecg_level": [],
        "cxr_level": [],
        "modality_gate": [],
        "attn_expert_gate": [],
    }

    for x_ecg, x_cxr, x_tab, y in loader:
        x_ecg = x_ecg.to(DEVICE, non_blocking=True)
        x_cxr = x_cxr.to(DEVICE, non_blocking=True)
        x_tab = x_tab.to(DEVICE, non_blocking=True)

        out = model(x_ecg, x_cxr, x_tab)

        ys.append(y.cpu().numpy())
        p_final_list.append(out["p_final"].cpu().numpy())
        p_m1_list.append(out["p_m1"].cpu().numpy())
        p_m2_list.append(out["p_m2"].cpu().numpy())

        gate_records["ehr_level"].append(out["selected"]["EHR"]["level_gate"].cpu().numpy())
        gate_records["ecg_level"].append(out["selected"]["ECG"]["level_gate"].cpu().numpy())
        gate_records["cxr_level"].append(out["selected"]["CXR"]["level_gate"].cpu().numpy())
        gate_records["modality_gate"].append(out["modality_gate"].cpu().numpy())
        gate_records["attn_expert_gate"].append(out["attn_expert_gate"].cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    p_final = np.concatenate(p_final_list, axis=0)
    p_m1 = np.concatenate(p_m1_list, axis=0)
    p_m2 = np.concatenate(p_m2_list, axis=0)

    gates = {k: np.concatenate(v, axis=0) for k, v in gate_records.items()}
    return y_true, {"Final": p_final, "Module1": p_m1, "Module2": p_m2}, gates


@torch.no_grad()
def evaluate(model, loader, return_loss=False, class_weights=None):
    y_true, probas, gates = predict_all(model, loader)

    metrics = {
        name: get_clf_eval_mc(name, y_true, p, elapsed_sec=0.0)
        for name, p in probas.items()
    }

    if not return_loss:
        return metrics

    total_loss = 0.0
    n = 0
    for x_ecg, x_cxr, x_tab, y in loader:
        x_ecg = x_ecg.to(DEVICE, non_blocking=True)
        x_cxr = x_cxr.to(DEVICE, non_blocking=True)
        x_tab = x_tab.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        out = model(x_ecg, x_cxr, x_tab)
        loss = F.cross_entropy(
            out["z_final"], y,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return metrics, total_loss / max(n, 1), gates



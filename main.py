# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pandas as pd
import torch

from config import *
from dataset import create_repeat_loaders, prepare_data
from fusion_model import TrimodalMoE
from model_builders import build_cxr_model, build_ecg_model, build_tab_model
from train_utils import predict_all, train_one_model
from utils import (
    get_clf_eval_mc, make_experiment_name, save_sample_level_outputs,
    set_seed, summarize_gates,
)

def main():
    set_seed(SEED_BASE)

    x_ecg_all, cxr_paths, X_tab_all, y_all, num_features, cat_features, cat_dims = prepare_data()
    num_cnt = len(num_features)
    tab_dim = X_tab_all.shape[1]

    total_combinations = len(GATE_KL_WEIGHT_LIST) * len(GUMBEL_TAU_LIST)
    total_training_runs = total_combinations * N_REPEATS

    print("\n" + "=" * 90)
    print("Gate KL x Gumbel Tau grid search")
    print(f"Gate KL weights : {GATE_KL_WEIGHT_LIST}")
    print(f"Gumbel taus     : {GUMBEL_TAU_LIST}")
    print(f"Grid combinations: {total_combinations}")
    print(f"Repeats per combination: {N_REPEATS}")
    print(f"Total training runs: {total_training_runs}")
    print("=" * 90)

    grid_config_rows = []
    all_results = []
    all_gate_summaries = []
    all_best_rows = []

    run_index = 0

    # Repeat is outside the grid loops so every hyperparameter combination
    # uses exactly the same split for a given repeat.
    for repeat in range(N_REPEATS):
        seed = SEED_BASE + repeat
        set_seed(seed)

        print(
            f"\n{'#' * 90}\n"
            f"REPEAT {repeat + 1}/{N_REPEATS} | seed={seed}\n"
            f"{'#' * 90}"
        )

        dl_train, dl_val, dl_test = create_repeat_loaders(
            x_ecg_all=x_ecg_all,
            cxr_paths=cxr_paths,
            x_tab_all=X_tab_all,
            y_all=y_all,
            num_count=num_cnt,
            seed=seed,
        )

        for gate_kl_weight in GATE_KL_WEIGHT_LIST:
            for gumbel_tau in GUMBEL_TAU_LIST:
                run_index += 1

                experiment_name = make_experiment_name(
                    gate_kl_weight=gate_kl_weight,
                    gumbel_tau=gumbel_tau,
                )
                experiment_dir = os.path.join(OUT_DIR, experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)

                grid_config_rows.append({
                    "Experiment": experiment_name,
                    "Gate_KL_Weight": float(gate_kl_weight),
                    "Gumbel_Tau": float(gumbel_tau),
                    "Repeat": repeat,
                    "Seed": seed,
                    "Output_Dir": experiment_dir,
                })

                print(
                    f"\n{'=' * 90}\n"
                    f"GRID RUN {run_index}/{total_training_runs} | "
                    f"Repeat={repeat + 1}/{N_REPEATS} | "
                    f"GateKL={gate_kl_weight:g} | Tau={gumbel_tau:g}\n"
                    f"{'=' * 90}"
                )

                # Re-seed before every grid run for a controlled comparison.
                # The same repeat uses the same initialization seed across combinations.
                set_seed(seed)

                for ecg_name in ECG_MODEL_NAMES:
                    for tab_name in TAB_MODEL_NAMES:
                        for cxr_name in CXR_MODEL_NAMES:
                            combo_name = (
                                f"LeMoFMoE_ECG-{ecg_name}"
                                f"_EHR-{tab_name}_CXR-{cxr_name}"
                            )
                            print(f"\n[Train] {combo_name}")

                            ecg_encoder = build_ecg_model(ecg_name)
                            ehr_encoder = build_tab_model(
                                tab_name, tab_dim, num_cnt, cat_dims
                            )
                            cxr_encoder = build_cxr_model(cxr_name)

                            model = TrimodalLeMoFMoE(
                                ecg_encoder=ecg_encoder,
                                ehr_encoder=ehr_encoder,
                                cxr_encoder=cxr_encoder,
                                d_model=D_MODEL,
                                gumbel_tau=gumbel_tau,
                            )

                            t0 = time.time()
                            model, hist_df, best_epoch, best_auc = train_one_model(
                                model=model,
                                train_loader=dl_train,
                                val_loader=dl_val,
                                epochs=EPOCHS,
                                patience=PATIENCE,
                                use_early_stopping=USE_EARLY_STOPPING,
                                gate_kl_weight=gate_kl_weight,
                                gumbel_tau=gumbel_tau,
                            )
                            elapsed = time.time() - t0

                            history_path = os.path.join(
                                experiment_dir,
                                f"history_repeat{repeat}_{combo_name}.csv",
                            )
                            hist_df.to_csv(history_path, index=False)

                            y_true, probas, gates = predict_all(model, dl_test)

                            sample_output_path = os.path.join(
                                experiment_dir,
                                f"sample_outputs_repeat{repeat}_{combo_name}.csv",
                            )
                            save_sample_level_outputs(
                                save_path=sample_output_path,
                                y_true=y_true,
                                probas=probas,
                                gates=gates,
                                gate_kl_weight=gate_kl_weight,
                                gumbel_tau=gumbel_tau,
                            )

                            for stage_name, probability in probas.items():
                                row = get_clf_eval_mc(
                                    f"{combo_name}_{stage_name}",
                                    y_true,
                                    probability,
                                    elapsed_sec=elapsed,
                                )
                                row.update({
                                    "Experiment": experiment_name,
                                    "Gate_KL_Weight": float(gate_kl_weight),
                                    "Gumbel_Tau": float(gumbel_tau),
                                    "Repeat": repeat,
                                    "Seed": seed,
                                    "Stage": stage_name,
                                    "ECG_Model": ecg_name,
                                    "EHR_Model": tab_name,
                                    "CXR_Model": cxr_name,
                                    "Best_Epoch": int(best_epoch),
                                    "Best_Val_AUC": float(best_auc),
                                })
                                all_results.append(row)

                                print(
                                    f"[{stage_name}] "
                                    f"Acc={row['Accuracy']:.5f}, "
                                    f"BAcc={row['Balanced_Acc']:.5f}, "
                                    f"F1={row['Macro_F1']:.5f}, "
                                    f"AUC={row['Macro_ROC_AUC(ovr)']:.5f}"
                                )

                            all_best_rows.append({
                                "Experiment": experiment_name,
                                "Gate_KL_Weight": float(gate_kl_weight),
                                "Gumbel_Tau": float(gumbel_tau),
                                "Repeat": repeat,
                                "Seed": seed,
                                "Model": combo_name,
                                "Best_Epoch": int(best_epoch),
                                "Best_Val_AUC": float(best_auc),
                                "Elapsed_sec": float(elapsed),
                            })

                            gate_df = summarize_gates(
                                gates=gates,
                                repeat=repeat,
                                model_name=combo_name,
                                gate_kl_weight=gate_kl_weight,
                                gumbel_tau=gumbel_tau,
                            )
                            gate_df["Experiment"] = experiment_name
                            all_gate_summaries.append(gate_df)

                            checkpoint_path = os.path.join(
                                experiment_dir,
                                f"checkpoint_repeat{repeat}_{combo_name}.pt",
                            )
                            torch.save({
                                "model_state_dict": model.state_dict(),
                                "config": {
                                    "experiment": experiment_name,
                                    "ecg_model": ecg_name,
                                    "ehr_model": tab_name,
                                    "cxr_model": cxr_name,
                                    "d_model": D_MODEL,
                                    "attn_len": ATTN_LEN,
                                    "n_classes": N_CLASSES,
                                    "epochs": EPOCHS,
                                    "use_early_stopping": USE_EARLY_STOPPING,
                                    "patience": PATIENCE,
                                    "early_stopping_metric": "validation_final_auc",
                                    "gate_kl_weight": float(gate_kl_weight),
                                    "gumbel_tau": float(gumbel_tau),
                                    "label_smoothing": LABEL_SMOOTHING,
                                    "best_epoch": int(best_epoch),
                                    "best_val_auc": float(best_auc),
                                    "module1_gate_input_norm": "LayerNorm",
                                    "module1_internal_fusion": "weighted_logits",
                                    "module2_ehr_enhancement":
                                        "EHR_residual_after_cross_modal_fusion",
                                    "module2_pooling":
                                        "learnable_attention_pooling",
                                    "module2_internal_fusion":
                                        "weighted_logits",
                                    "final_fusion":
                                        "concat_module_logits_then_mlp",
                                },
                            }, checkpoint_path)

                            del model, ecg_encoder, ehr_encoder, cxr_encoder
                            torch.cuda.empty_cache()

                # Save cumulative progress after every grid run.
                pd.DataFrame(all_results).to_csv(
                    os.path.join(OUT_DIR, "grid_results_live.csv"),
                    index=False,
                )
                pd.DataFrame(all_best_rows).to_csv(
                    os.path.join(OUT_DIR, "grid_best_validation_live.csv"),
                    index=False,
                )
                if all_gate_summaries:
                    pd.concat(
                        all_gate_summaries,
                        axis=0,
                        ignore_index=True,
                    ).to_csv(
                        os.path.join(OUT_DIR, "grid_gate_summary_live.csv"),
                        index=False,
                    )

    # --------------------------------------------------------
    # Final global summaries
    # --------------------------------------------------------
    config_df = pd.DataFrame(grid_config_rows)
    config_df.to_csv(
        os.path.join(OUT_DIR, "grid_config.csv"),
        index=False,
    )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(OUT_DIR, "trimodal_lemof_moe_grid_results.csv"),
        index=False,
    )

    best_df = pd.DataFrame(all_best_rows)
    best_df.to_csv(
        os.path.join(OUT_DIR, "grid_best_validation.csv"),
        index=False,
    )

    if all_gate_summaries:
        gate_summary_df = pd.concat(
            all_gate_summaries,
            axis=0,
            ignore_index=True,
        )
        gate_summary_df.to_csv(
            os.path.join(OUT_DIR, "trimodal_lemof_moe_grid_gate_summary.csv"),
            index=False,
        )

    # Mean/std test summary across repeats for each hyperparameter combination.
    if not results_df.empty:
        metric_cols = [
            "Accuracy",
            "Balanced_Acc",
            "Macro_Precision",
            "Macro_Recall",
            "Macro_F1",
            "Macro_ROC_AUC(ovr)",
            "Elapsed_sec",
        ]

        aggregate_df = (
            results_df
            .groupby(
                [
                    "Experiment",
                    "Gate_KL_Weight",
                    "Gumbel_Tau",
                    "Stage",
                    "ECG_Model",
                    "EHR_Model",
                    "CXR_Model",
                ],
                as_index=False,
            )[metric_cols]
            .agg(["mean", "std"])
        )

        # Flatten multi-index columns.
        aggregate_df.columns = [
            "_".join(
                str(part) for part in column if str(part) != ""
            ).rstrip("_")
            if isinstance(column, tuple)
            else str(column)
            for column in aggregate_df.columns
        ]

        aggregate_df.to_csv(
            os.path.join(OUT_DIR, "grid_results_mean_std.csv"),
            index=False,
        )

        final_only = results_df[results_df["Stage"] == "Final"].copy()
        if not final_only.empty:
            ranking_df = (
                final_only
                .groupby(
                    ["Experiment", "Gate_KL_Weight", "Gumbel_Tau"],
                    as_index=False,
                )
                .agg(
                    Test_AUC_Mean=("Macro_ROC_AUC(ovr)", "mean"),
                    Test_AUC_Std=("Macro_ROC_AUC(ovr)", "std"),
                    Test_F1_Mean=("Macro_F1", "mean"),
                    Test_F1_Std=("Macro_F1", "std"),
                    Test_BAcc_Mean=("Balanced_Acc", "mean"),
                    Test_BAcc_Std=("Balanced_Acc", "std"),
                    Test_Accuracy_Mean=("Accuracy", "mean"),
                    Test_Accuracy_Std=("Accuracy", "std"),
                    Mean_Best_Val_AUC=("Best_Val_AUC", "mean"),
                    Mean_Best_Epoch=("Best_Epoch", "mean"),
                )
                .sort_values(
                    ["Mean_Best_Val_AUC", "Test_AUC_Mean"],
                    ascending=False,
                )
            )
            ranking_df.insert(
                0,
                "Rank_by_Validation_AUC",
                np.arange(1, len(ranking_df) + 1),
            )
            ranking_df.to_csv(
                os.path.join(OUT_DIR, "grid_final_ranking.csv"),
                index=False,
            )

            print("\nTop grid combinations ranked by mean best validation AUC:")
            print(ranking_df.head(10).to_string(index=False))

    print("\nAll grid tasks completed.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

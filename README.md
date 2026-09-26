# Refactored trimodal LeMoF-MoE experiment

## Structure
- `config.py`: fixed constants and paths
- `dataset.py`: loading, preprocessing, split, scaling, DataLoaders
- `model_builders.py`: ECG/EHR/CXR encoder builders
- `fusion_model.py`: level routing, modality MoE, interaction MoE, final fusion
- `fusion_modsl.py`: compatibility alias for the requested typo filename
- `train_utils.py`: training, prediction, evaluation
- `utils.py`: metrics, seeds, gate summaries, CSV output helpers
- `main.py`: experiment orchestration
- `head.py`: compatibility wrapper around the original `heads.py`

## Model folders
The requested ECG folder name `model_ccg` is preserved. Put these existing files inside it:
- `ecg_baseline.py`
- `ecg_resnet.py`
- `wavenet.py`
- `lstm.py`

Put the existing EHR files inside `model_ehr`:
- `tab_baseline.py`
- `tpc.py`
- `tabnet.py`
- `tabtransformer.py`
- `fttransformer.py`

`model_cxr` is included for future CXR encoder files. The current ResNet18 implementation remains in `model_builders.py` so the behavior is preserved.

Place the original `heads.py` beside `head.py`, unless you move `BaseModel` and `HeadBlock` directly into `head.py`.

Run:
```bash
python main.py
```

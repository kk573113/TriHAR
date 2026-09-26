# Refactored trimodal framework 'TriHAR' experiment

## Structure
- `config.py`: fixed constants and paths
- `dataset.py`: loading, preprocessing, split, scaling, DataLoaders
- `model_builders.py`: ECG/EHR/CXR encoder builders
- `fusion_model.py`: level routing, modality MoE, interaction MoE, final fusion
- `train_utils.py`: training, prediction, evaluation
- `utils.py`: metrics, seeds, gate summaries, CSV output helpers
- `main.py`: experiment orchestration
- `heads.py`: compatibility wrapper

## Model folders
Put these existing files inside it `model_ecg`:
- `ecg_resnet.py`
- `wavenet.py`
- `lstm.py`

Put the existing EHR files inside `model_ehr`:
- `tpc.py`
- `tabtransformer.py`
- `fttransformer.py`

Put the existing EHR files inside `model_cxr`:
- `DenseNet121.py`
- `VGG16.py`
- `resnet18.py`


Run:
```bash
python main.py
```

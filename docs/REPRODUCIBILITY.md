# Reproducibility guide

This document describes how to work with the current notebook release and distinguishes the paper protocol from repository-specific details.

## 1. What is included

| Notebook | Primary purpose | Framework style |
|---|---|---|
| `Code/dgmp-net.ipynb` | TLGAN generation, purifier construction, and DGMP-Net evaluation | TensorFlow/Keras research notebook |
| `Code/uman-net.ipynb` | NUMAN/LUMAN modules, UMAN-Net training, and adversarial evaluation | TensorFlow/Keras + CleverHans |
| `Code/mmifan-net.ipynb` | Multimodal data preparation, MMIFAN modules, fusion training, and attack evaluation | TensorFlow/Keras + CleverHans |

The notebooks record Python 3.10.13 in their metadata and were authored in GPU-enabled Kaggle-style environments.

## 2. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

The provided dependency file is a compatibility baseline reconstructed from notebook imports, not an archival lockfile. For a formal reproduction, record the fully resolved environment with:

```bash
pip freeze > environment-resolved.txt
```

## 3. Dataset-path adaptation

The notebooks preserve authoring-time path cells. Examples include:

- DGMP-Net: `/input/chest-xray-pneumonia/chest_xray/train` and `/test`.
- UMAN-Net: a local `sarscov2-ctscan-dataset/{COVID,non-COVID}` directory.
- MMIFAN-Net: Kaggle-style NumPy arrays under `/input/updated-sars-covid-ct-scan-dataset/...` and `/input/pulmonary-dataset/...`.

Before execution, replace these paths with locations on your system. Keep class ordering consistent with the notebook's label list and verify array shapes before training.

## 4. Paper training protocol

| Setting | Value reported in the paper |
|---|---|
| Epochs | 200 |
| Batch size | 32 |
| Objective | Cross-entropy for classification; task-specific segmentation losses in the segmentation backbones |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| LR schedule | ReduceLROnPlateau, factor 0.1 after 10 epochs without improvement, minimum LR 1e-6 |
| Hardware | NVIDIA GeForce RTX 4060 Ti |
| Post-training protocol | Freeze trained model weights before adversarial evaluation |

The notebooks may contain shorter exploratory runs or dataset-specific settings. For paper-level reproduction, align notebook cells with the configuration above.

## 5. Attack protocols

### Unimodal classification with VGG16

- Attacks: FGSM, PGD-20, BIM-20, and MIM-20.
- Perturbation budgets: ε ∈ {0.3, 0.6, 0.9, 1.2}.
- PGD step size: ε/4.
- BIM/MIM step size: ε/10.

### Unimodal classification with ResNet18/ResNet50

- Attacks: FGSM, PGD-100, BIM-100, and MIM-100.
- Perturbation budgets: ε ∈ {8, 16, 24, 32}/255.
- Iterative attacks use 100 steps.

### Multimodal classification

- Main attacks: PGD-20 and MIM-20.
- Perturbation budgets: 8/255 to 32/255 in 4/255 increments.
- Stronger PGD sweeps: 10–100 iterations and ε from 0.01 to 0.1.
- Additional evaluations: Admix and DeCoWA.

### Segmentation

- Backbones: PraNet and MADGNet.
- Attacks: FGSM, PGD-100, BIM-100, and MIM-100.
- Perturbation budgets: ε ∈ {8, 16, 24, 32}/255.
- Perturbations are mask-constrained: gradients are zeroed within lesion/organ masks so that updates affect non-lesion pixels.

## 6. Proposed-method training policy

The paper's proposed DGMP-Net, UMAN-Net, and MMIFAN-Net are evaluated **without an adversarial-training phase**. AT, PNI, and DBN-AT baselines use PGD-based adversarial training. Do not silently add adversarial training to the proposed methods when attempting to match the reported protocol.

## 7. Determinism and result matching

For a rigorous reproduction:

1. Fix Python, NumPy, TensorFlow, and data-split seeds.
2. Save exact train/validation/test manifests rather than regenerating splits.
3. Record dataset checksums and preprocessing versions.
4. Disable nondeterministic GPU operations where feasible.
5. Save the best validation checkpoint and the final frozen model separately.
6. Report clean and adversarial metrics for every attack/budget pair.
7. Average runtime over a fixed warm-up and measurement protocol.

The current repository does not include the original split manifests or pretrained checkpoints, so small deviations from paper tables should be expected unless the original experimental state is reconstructed.

## 8. Recommended output organization

```text
outputs/
├── dgmp/
│   ├── checkpoints/
│   ├── reconstructions/
│   └── metrics.csv
├── uman/
│   ├── checkpoints/
│   ├── attacks/
│   └── metrics.csv
└── mmifan/
    ├── checkpoints/
    ├── attacks/
    └── metrics.csv
```

The `outputs/` directory is ignored by Git so that local checkpoints and protected medical data are not committed accidentally.

## 9. Reporting checklist

A result should identify:

- dataset and exact version;
- task and backbone;
- method variant;
- clean accuracy or Dice;
- attack, iteration count, ε, step size, and restart policy;
- whether the evaluation is white-box;
- random seed and number of runs;
- parameter count, FLOPs convention, and timing hardware.

For complete tables and ablations, consult the article and supplementary material: <https://doi.org/10.1016/j.inffus.2025.103822>.

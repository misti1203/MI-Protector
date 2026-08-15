# Dataset inventory

The paper evaluates MI-Protector on **16 publicly available medical-imaging datasets**. The table below reproduces the names, modalities, sample counts, and class counts reported in Appendix Table 1. D1–D14 are used for classification analyses, while D15–D16 are used for segmentation robustness experiments.

| ID | Dataset | Modality | Task in paper | Images | Classes |
|---:|---|---|---|---:|---:|
| D1 | CXR Pneumonia | Chest X-ray | Classification | 5,863 | 2 |
| D2 | COVID-19 CXR | Chest X-ray | Classification | 1,823 | 3 |
| D3 | SARS-CoV-2 | CT scan | Classification | 2,481 | 2 |
| D4 | Lung Cancer | CT scan | Classification | 1,097 | 3 |
| D5 | Brain 35H | MRI | Classification | 3,000 | 2 |
| D6 | Breast Cancer | Ultrasound | Classification | 1,578 | 3 |
| D7 | Kaggle Skin Cancer | Dermoscopy | Classification | 3,297 | 2 |
| D8 | Wireless Capsule Endoscopy (WCE) | Endoscopy | Classification | 6,000 | 4 |
| D9 | SIPaKMeD | Pap smear | Classification | 4,049 | 5 |
| D10 | C-NMC 2019 | Microscopy cell images | Classification | 15,135 | 2 |
| D11 | Pulmonary CXR Abnormalities | Chest X-ray | Classification | 662 | 2 |
| D12 | OrganAMNIST | Abdominal CT | Classification | 58,830 | 11 |
| D13 | PathMNIST | Colon pathology | Classification | 107,180 | 9 |
| D14 | TissueMNIST | Kidney-cortex microscopy | Classification | 236,386 | 8 |
| D15 | COVID-19 Lung CT | Chest CT | Segmentation | 812 | 2 |
| D16 | ISIC2018 | Dermoscopy | Segmentation | 10,015 | 7 |

## Evaluation groupings

- **Unimodal classification:** DGMP-Net and UMAN-Net are evaluated on selected datasets from D1–D11 with VGG16, ResNet18, and ResNet50.
- **Multimodal classification:** primary robustness tables use D1, D3, D5, and D11; stronger PGD analyses also use D12–D14.
- **Segmentation:** UMAN-integrated PraNet and MADGNet are evaluated on D15 and D16 with mask-constrained adversarial attacks.
- **Clinical noise validation:** Poisson-noise experiments use D3 and D4.

## Data policy

This repository does not redistribute medical images. Obtain each dataset from its original provider and comply with the corresponding license, access conditions, citation requirements, and privacy restrictions. Dataset versions can differ over time; record checksums and source dates when reproducing experiments.

## Preprocessing reported in the paper

1. Resize inputs to `128 × 128 × 3` for classification or `224 × 224 × 3` for segmentation.
2. Apply per-image min–max normalization to `[0, 1]`.
3. Split each dataset into `70%` training, `10%` validation, and `20%` testing.
4. During training, augment data with 20° rotation, ±5-pixel translation, and 3×3 Gaussian smoothing.
5. Preserve test data for adversarial-example generation and robustness evaluation.

For complete provenance and source references, consult the paper and its supplementary material: <https://doi.org/10.1016/j.inffus.2025.103822>.

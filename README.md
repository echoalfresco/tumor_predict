# tumor_predict

Binary classification of breast ultrasound images as **benign** or **malignant** with a fine-tuned DenseNet121, evaluated with duplicate-aware, grouped 5-fold cross-validation.

> **Educational project. Not a medical device.** Do not use these results or this model for diagnosis, screening, or any clinical decision.

## Dataset

- [BUSI-corrected](https://www.kaggle.com/datasets/jarintasnim090/busi-corrected) (Kaggle, by J. Tasnim), a corrected version of the widely used Breast Ultrasound Images (BUSI) dataset of Al-Dhabyani et al. (2020).
- The `normal` class and the segmentation masks are excluded. This project uses **589 images: 410 benign and 179 malignant**.
- Class labels come from the **folder name** (`benign/` or `malignant/`).

## Method

| Step | Detail |
|---|---|
| Model | DenseNet121, ImageNet-pretrained, new 2-class head |
| Input | 224x224 RGB; flip, +/-10 degree rotation, brightness/contrast jitter during training |
| Training | AdamW (lr 1e-4, weight decay 1e-4), cosine schedule, class-weighted cross-entropy, up to 15 epochs, best epoch chosen by validation AUC |
| Duplicate control | Perceptual hashing (pHash, Hamming distance <= 4) groups near-duplicates: **490 groups from 589 images**. Groups are never split across train, validation, and test |
| Evaluation | `StratifiedGroupKFold`, 5 folds. Inside each fold, about 15% of the training pool (by group) is the validation set for epoch selection and threshold tuning |
| Threshold tuning | Per fold, the highest cutoff (capped at 0.5) that reaches 90% sensitivity on that fold's validation set, then applied to the held-out fold |

Every image receives exactly one out-of-fold prediction, so all metrics below pool the 589 held-out predictions.

## Results

### Cross-validated (5-fold, 589 images)

| Operating point | Sensitivity (malignant) | Specificity | Missed malignant | False alarms |
|---|---|---|---|---|
| Threshold 0.5 | 0.838 (95% CI 0.78-0.88) | 0.937 (0.91-0.96) | 29 | 26 |
| Tuned per-fold threshold | 0.883 (0.83-0.92) | 0.912 (0.88-0.94) | 21 | 36 |

- **Pooled out-of-fold AUC: 0.963.** Per-fold test AUC: mean 0.968, SD 0.016 (range 0.946-0.991).
- Accuracy: 0.907 at threshold 0.5, 0.903 with tuned thresholds.
- Confidence intervals are 95% Wilson intervals.

| Fold | n test | Val AUC | Test AUC | Tuned threshold | Sens @0.5 | Spec @0.5 | Sens tuned | Spec tuned |
|---|---|---|---|---|---|---|---|---|
| 0 | 115 | 0.981 | 0.972 | 0.378 | 0.846 | 0.974 | 0.923 | 0.934 |
| 1 | 120 | 0.982 | 0.967 | 0.251 | 0.838 | 0.892 | 0.892 | 0.855 |
| 2 | 124 | 0.994 | 0.962 | 0.500 | 0.800 | 0.910 | 0.800 | 0.910 |
| 3 | 116 | 0.986 | 0.946 | 0.500 | 0.833 | 0.912 | 0.833 | 0.912 |
| 4 | 114 | 0.963 | 0.991 | 0.300 | 0.875 | 1.000 | 0.969 | 0.951 |

### Single 81-image hold-out runs (for reference)

Two single-split runs (15 epochs each) gave test AUC 0.960 and 0.977, accuracy 0.914 and 0.938, and malignant sensitivity 0.80 and 0.84. Those splits contain only 25 malignant test images, so the cross-validated numbers above are the more reliable estimate.

### Error analysis

- Filenames and folders disagree for **27 images** (19 files named `malignant ...` in `benign/`, 8 named `benign ...` in `malignant/`), possibly cases relabeled in the corrected release (unconfirmed).
- At threshold 0.5 these 27 images have a **48% error rate (13 errors)**, versus **7.5% (42 errors)** on the other 562 images. They account for 24% of all errors while making up 4.6% of the data.
- On the 562 images where folder and filename agree: sensitivity 0.865, specificity 0.951, accuracy 0.925, AUC 0.971. This subset figure is post hoc and should not be read as the headline result.
- Sweeping a single global threshold over the pooled predictions (post hoc, so optimistic): 0.4 gives sensitivity 0.883 / specificity 0.920; 0.3 gives 0.922 / 0.871; 0.2 gives 0.944 / 0.817.

## Limitations

- **Not clinically validated.** Single public dataset, retrospective, no prospective data, and no comparison against radiologists.
- **No patient-level split.** Patient IDs are not available here. Grouping uses perceptual-hash similarity, which removes near-duplicates but cannot guarantee that images from the same patient stay on one side of a split. Residual leakage would make results optimistic.
- **Sensitivity is below screening standards.** Roughly 1 in 6 malignant lesions is missed at threshold 0.5, and about 1 in 9 with tuned thresholds. Missed cancers are the costly error, and raising sensitivity costs specificity (see the threshold sweep).
- **Label uncertainty.** Labels are taken from folder names. 27 images have a filename that contradicts the folder, and 4 malignant and 3 benign images are misclassified with high confidence (predicted probability below 0.1 or above 0.9). Some of these may be label errors rather than model errors. No expert re-review has been done.
- **Threshold tuning is noisy.** Each fold's validation set holds only about 25 malignant cases, so per-fold thresholds vary (0.25 to 0.50). A single global threshold chosen on independent data would be more stable. The post-hoc global sweep above uses the same predictions it is evaluated on.
- **Small, single-source data.** 589 images, 179 malignant, one imaging source. Published work shows CNNs trained on BUSI lose substantial performance on other institutions' data, so expect weaker results on any other scanner or population.
- **Dataset-quality caveats.** The original BUSI set contains many duplicate and inconsistently labeled images. Accuracy figures in papers that do not deduplicate (often 94-99%) are not comparable to the grouped-CV figures here.
- **No explainability.** There is no saliency or Grad-CAM analysis, so it is unknown whether the model attends to the lesion or to image artifacts such as annotations or scanner overlays.
- **Single architecture and seed.** Only DenseNet121 with seed 42 was tried; no hyperparameter search or ensembling.

## Reproducing

1. Kaggle notebook with GPU and Internet enabled. Use *Save & Run All (Commit)* to avoid inactivity timeouts.
2. `pip install kagglehub torch torchvision scikit-learn pillow imagehash pandas`
3. Run `busi_densenet121.py` (the 5-fold run took about 6 minutes on a Kaggle GPU).

Outputs: `busi_oof_predictions.csv` (one prediction per image), `busi_fold_metrics.csv` (per-fold table), `busi_misclassified.csv` (errors at 0.5, for manual review).

## Possible next steps

- Choose one global threshold from pooled validation predictions, then evaluate it on untouched data.
- Have an expert review the 27 mismatched images and the confident errors, and report results with and without them.
- Add Grad-CAM, repeat with several seeds, and try stronger augmentation or other backbones.
- Validate on an independent ultrasound dataset.

## Citations

If you use the dataset, the corrected-dataset authors ask that you cite:

> J. Tasnim and M. K. Hasan, "CAM-QUS guided self-tuning modular CNNs with multi-loss functions for fully automated breast lesion classification in ultrasound images," *Physics in Medicine & Biology*, 2023. https://doi.org/10.1088/1361-6560/ad1319

The corrected set is derived from the original BUSI dataset:

> Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. *Data in Brief*. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863

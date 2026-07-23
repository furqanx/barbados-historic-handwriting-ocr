# Kraken Historical HTR Workflow

Kraken is handled as an external historical HTR workflow. The project converts
our line-level manifests into Kraken's legacy line-strip format, fine-tunes a
`.mlmodel` with `ketos train`, recognizes prepared line images with `kraken`,
and converts plain text outputs back to prediction/submission CSV files.

Useful official references:

- https://kraken.re/6.0.0/training/rectrain.html
- https://kraken.re/6.0.0/advanced/repo.html
- https://kraken.re/6.0.0/index.html

## Setup

```bash
!git -C /kaggle/working/barbados-historic-handwriting-ocr pull
!pip install -r /kaggle/working/barbados-historic-handwriting-ocr/requirements-kraken.txt
```

Kraken can require system libraries depending on the runtime. If Kaggle install
fails, move this branch of experiments to Colab, RunPod, Lambda, or another
Linux runtime where system packages are easier to control.

## Prepare Dataset

CATMuS and McCATMuS use NFD normalization, so start with `NFD` for those models.
TRIDIS uses a more standardized transcription convention; try `preserve` first.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/prepare_kraken_dataset.py \
  --train-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv \
  --test-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/test_manifest.csv \
  --output-dir /kaggle/working/barbados-historic-handwriting-ocr/data/kraken/fold0_h128_nfd \
  --fold 0 \
  --image-height 128 \
  --unicode-normalization NFD
```

## Audit Text Characters

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/audit_kraken_text.py \
  --train-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv \
  --unicode-normalization NFD
```

## Download Base Model

Start with CATMuS Medieval.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/download_kraken_model.py \
  --model-key catmus-medieval \
  --output-dir /kaggle/working/barbados-historic-handwriting-ocr/models/kraken
```

Other supported keys:

```text
mccatmus
tridis
```

## Fine-Tune

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_kraken.py \
  --run-name kraken_catmus_fold0_h128_nfd \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/kraken/fold0_h128_nfd \
  --base-model /kaggle/working/barbados-historic-handwriting-ocr/models/kraken/catmus-medieval/catmus-medieval-1.6.0.mlmodel \
  --epochs 50 \
  --min-epochs 10 \
  --lag 10 \
  --batch-size 8 \
  --lr 3e-4 \
  --resize new \
  --unicode-normalization NFD \
  --device cuda:0 \
  --precision bf16-mixed
```

If a Kraken version rejects a CLI argument, inspect `ketos train --help`, then
pass the corrected runtime-specific argument with repeated `--extra-arg`.

## Ketos Validation Report

After training, point `--model` to the best `.mlmodel` saved by Kraken in
`outputs/kraken/<run_name>/`.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/test_kraken.py \
  --run-name kraken_catmus_fold0_h128_nfd \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/kraken/fold0_h128_nfd \
  --model /kaggle/working/barbados-historic-handwriting-ocr/outputs/kraken/kraken_catmus_fold0_h128_nfd/kraken_catmus_fold0_h128_nfd_best.mlmodel \
  --split val \
  --unicode-normalization NFD \
  --device cuda:0
```

## Predict Test

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/predict_kraken.py \
  --run-name kraken_catmus_fold0_h128_nfd \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/kraken/fold0_h128_nfd \
  --model /kaggle/working/barbados-historic-handwriting-ocr/outputs/kraken/kraken_catmus_fold0_h128_nfd/kraken_catmus_fold0_h128_nfd_best.mlmodel \
  --split test
```

## Convert To Submission

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/convert_kraken_predictions.py \
  --run-name kraken_catmus_fold0_h128_nfd \
  --split test \
  --prediction-dir /kaggle/working/barbados-historic-handwriting-ocr/outputs/kraken/kraken_catmus_fold0_h128_nfd/test_raw \
  --sample-submission "/kaggle/input/datasets/furqanalghifari/historic-handwriting-comp/raw/road-barbados-historic-handwriting-challenge/SampleSubmission.csv" \
  --output-unicode-normalization NFC
```

## Predict Validation To Local WER/CER

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/predict_kraken.py \
  --run-name kraken_catmus_fold0_h128_nfd \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/kraken/fold0_h128_nfd \
  --model /kaggle/working/barbados-historic-handwriting-ocr/outputs/kraken/kraken_catmus_fold0_h128_nfd/kraken_catmus_fold0_h128_nfd_best.mlmodel \
  --split val
```

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/convert_kraken_predictions.py \
  --run-name kraken_catmus_fold0_h128_nfd \
  --split val \
  --prediction-dir /kaggle/working/barbados-historic-handwriting-ocr/outputs/kraken/kraken_catmus_fold0_h128_nfd/val_raw \
  --reference-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv \
  --output-unicode-normalization NFC
```

## Files To Download

Before stopping a cloud session, save:

```text
outputs/kraken/
outputs/predictions/
outputs/submissions/
data/kraken/
models/kraken/
```


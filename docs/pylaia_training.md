# PyLaia HTR Workflow

PyLaia is handled as an external HTR workflow: prepare a PyLaia dataset, download
a Teklia checkpoint, fine-tune with `pylaia-htr-train-ctc`, decode with
`pylaia-htr-decode-ctc`, then convert the raw decode output back to the
competition CSV format.

Official PyLaia docs describe the required dataset files (`train.txt`,
`val.txt`, `*_ids.txt`, `syms.txt`) and CLI entry points:

- https://doc.teklia.com/pylaia/usage/datasets/format/
- https://doc.teklia.com/pylaia/usage/training/
- https://doc.teklia.com/pylaia/usage/prediction/

## Kaggle Setup

PyLaia 1.1.2 officially supports Python 3.9 and 3.10. If a cloud runtime uses
Python 3.11+, install/run this workflow inside a Python 3.10 environment.

```bash
!git -C /kaggle/working/barbados-historic-handwriting-ocr pull
!pip install -r /kaggle/working/barbados-historic-handwriting-ocr/requirements-pylaia.txt
```

## Prepare Dataset

This resizes line images to fixed height 128 while preserving aspect ratio, then
writes PyLaia text tables and `syms.txt`.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/prepare_pylaia_dataset.py \
  --train-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv \
  --test-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/test_manifest.csv \
  --output-dir /kaggle/working/barbados-historic-handwriting-ocr/data/pylaia/fold0_h128 \
  --fold 0 \
  --image-height 128
```

## Validate Dataset

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/validate_pylaia_dataset.py \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/pylaia/fold0_h128 \
  --run-name pylaia_fold0_h128_validation \
  --image-height 128
```

## Download Base Model

Start with Himanis.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/download_pylaia_model.py \
  --model-key himanis \
  --output-dir /kaggle/working/barbados-historic-handwriting-ocr/models/pylaia
```

Other supported keys: `belfort`, `norhand-v1`, `norhand-v3`, `iam`.

## Audit Base Charset

This is optional, but useful before fine-tuning a pretrained PyLaia checkpoint.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/audit_pylaia_symbols.py \
  --train-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv \
  --syms /kaggle/working/barbados-historic-handwriting-ocr/models/pylaia/pylaia-himanis/syms.txt
```

## Fine-Tune

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_pylaia.py \
  --run-name pylaia_himanis_fold0_h128 \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/pylaia/fold0_h128 \
  --base-model-dir /kaggle/working/barbados-historic-handwriting-ocr/models/pylaia/pylaia-himanis \
  --epochs 80 \
  --batch-size 16 \
  --lr 5e-4 \
  --gpus 1
```

If the installed PyLaia/PyTorch-Lightning version rejects `--trainer.gpus`, pass
runtime-specific trainer arguments through `--extra-arg`.

## Decode Test

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/predict_pylaia.py \
  --run-name pylaia_himanis_fold0_h128 \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/pylaia/fold0_h128 \
  --base-model-dir /kaggle/working/barbados-historic-handwriting-ocr/models/pylaia/pylaia-himanis \
  --split test \
  --batch-size 16
```

## Convert To Submission

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/convert_pylaia_predictions.py \
  --run-name pylaia_himanis_fold0_h128 \
  --split test \
  --raw-output /kaggle/working/barbados-historic-handwriting-ocr/outputs/pylaia/pylaia_himanis_fold0_h128/test_raw.txt \
  --sample-submission "/kaggle/input/datasets/furqanalghifari/historic-handwriting-comp/raw/road-barbados-historic-handwriting-challenge/SampleSubmission.csv"
```

## Decode Validation

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/predict_pylaia.py \
  --run-name pylaia_himanis_fold0_h128 \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/pylaia/fold0_h128 \
  --base-model-dir /kaggle/working/barbados-historic-handwriting-ocr/models/pylaia/pylaia-himanis \
  --split val \
  --batch-size 16
```

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/convert_pylaia_predictions.py \
  --run-name pylaia_himanis_fold0_h128 \
  --split val \
  --raw-output /kaggle/working/barbados-historic-handwriting-ocr/outputs/pylaia/pylaia_himanis_fold0_h128/val_raw.txt \
  --reference-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv
```

## Files To Download

Download these before stopping a cloud session:

```text
outputs/pylaia/
outputs/predictions/
outputs/submissions/
data/pylaia/
models/pylaia/
```

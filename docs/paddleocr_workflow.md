# PaddleOCR Recognition Workflow

PaddleOCR is handled as an external text recognition workflow. This branch is
less natural for this competition than CTC/TrOCR/PyLaia/Kraken because
PaddleOCR recognizers are often configured for shorter OCR crops, but it is
still useful as a controlled P3 experiment.

Official references:

- Recognition dataset format: https://www.paddleocr.ai/main/en/datasets/ocr_datasets.html
- Recognition training: https://paddlepaddle.github.io/PaddleOCR/v2.10.0/en/ppocr/model_train/recognition.html
- PaddleOCR install notes: https://www.paddleocr.ai/latest/en/quick_start.html

## Setup

Clone PaddleOCR next to this repo:

```bash
!git clone https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR
!git -C /kaggle/working/barbados-historic-handwriting-ocr pull
```

Install dependencies according to your CUDA runtime. For many Kaggle/Colab
runtimes you may need to choose the correct PaddlePaddle wheel manually instead
of blindly installing `paddlepaddle-gpu`.

```bash
!pip install -r /kaggle/working/barbados-historic-handwriting-ocr/requirements-paddleocr.txt
!pip install -r /kaggle/working/PaddleOCR/requirements.txt
```

## Prepare Dataset

This creates:

```text
rec_gt_train.txt
rec_gt_val.txt
test_images.txt
character_dict.txt
train/
val/
test/
```

Label rows follow PaddleOCR recognition format:

```text
image_path<TAB>target text
```

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/prepare_paddleocr_dataset.py \
  --train-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv \
  --test-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/test_manifest.csv \
  --output-dir /kaggle/working/barbados-historic-handwriting-ocr/data/paddleocr/fold0_h64 \
  --fold 0 \
  --image-height 64 \
  --image-format jpg \
  --path-mode absolute
```

## Audit Text Characters

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/audit_paddleocr_text.py \
  --train-manifest /kaggle/working/barbados-historic-handwriting-ocr/data/metadata/train_manifest.csv
```

## Choose Base Config

Pick a PaddleOCR recognition config from the cloned PaddleOCR repo, for example:

```text
/kaggle/working/PaddleOCR/configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml
```

For SVTR, choose an SVTR recognition config from `configs/rec/` if available in
the PaddleOCR version you cloned.

## Patch Config

Long handwritten lines need a larger image width and max text length than many
default PaddleOCR configs.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/make_paddleocr_config.py \
  --base-config /kaggle/working/PaddleOCR/configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
  --dataset-dir /kaggle/working/barbados-historic-handwriting-ocr/data/paddleocr/fold0_h64 \
  --run-name paddleocr_ppocrv3_h64_w2048_fold0 \
  --image-shape 3,64,2048 \
  --max-text-length 128 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --epoch-num 30 \
  --use-amp
```

## Train

Single GPU:

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_paddleocr_rec.py \
  --paddleocr-dir /kaggle/working/PaddleOCR \
  --config /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/paddleocr_ppocrv3_h64_w2048_fold0.yml
```

Multi-GPU:

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_paddleocr_rec.py \
  --paddleocr-dir /kaggle/working/PaddleOCR \
  --config /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/paddleocr_ppocrv3_h64_w2048_fold0.yml \
  --gpus "0,1"
```

## Evaluate

Point `--checkpoint` to the saved PaddleOCR checkpoint prefix, commonly
`best_accuracy`.

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/eval_paddleocr_rec.py \
  --paddleocr-dir /kaggle/working/PaddleOCR \
  --config /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/paddleocr_ppocrv3_h64_w2048_fold0.yml \
  --checkpoint /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/checkpoints/best_accuracy
```

## Predict Test

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/predict_paddleocr_rec.py \
  --paddleocr-dir /kaggle/working/PaddleOCR \
  --config /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/paddleocr_ppocrv3_h64_w2048_fold0.yml \
  --checkpoint /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/checkpoints/best_accuracy \
  --infer-img /kaggle/working/barbados-historic-handwriting-ocr/data/paddleocr/fold0_h64/test \
  --run-name paddleocr_ppocrv3_h64_w2048_fold0
```

## Convert To Submission

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/convert_paddleocr_predictions.py \
  --run-name paddleocr_ppocrv3_h64_w2048_fold0 \
  --split test \
  --raw-output /kaggle/working/barbados-historic-handwriting-ocr/outputs/paddleocr/paddleocr_ppocrv3_h64_w2048_fold0/test_raw.txt \
  --sample-submission "/kaggle/input/datasets/furqanalghifari/historic-handwriting-comp/raw/road-barbados-historic-handwriting-challenge/SampleSubmission.csv"
```

## Files To Download

Before stopping a cloud session:

```text
outputs/paddleocr/
outputs/predictions/
outputs/submissions/
data/paddleocr/
models/paddleocr/
```


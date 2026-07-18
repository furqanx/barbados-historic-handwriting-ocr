# Cloud Training Workflow

Use this repository as the source of truth. Edit code locally, push to GitHub,
then pull or reclone it inside Kaggle, Colab, RunPod, or Lambda.

## 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

On Kaggle/Colab, PyTorch is usually preinstalled. If dependency resolution is
slow, install only the missing packages.

## 2. Build manifests

Point these arguments to the dataset paths used by the runtime.

```bash
python scripts/make_folds.py \
  --train-csv /path/to/Train.csv \
  --test-csv /path/to/Test.csv \
  --sample-submission /path/to/SampleSubmission.csv \
  --image-dir /path/to/images \
  --train-output data/metadata/train_manifest.csv \
  --test-output data/metadata/test_manifest.csv
```

## 3. Build character vocabulary

```bash
python scripts/build_char_vocab.py \
  --train-csv /path/to/Train.csv \
  --output data/metadata/char_vocab.json
```

## 4. Train CRNN-CTC

Start with one fold and a named run.

```bash
python scripts/train_crnn_ctc.py \
  --run-name crnn_ctc_h96_w2048_fold0 \
  --fold 0 \
  --epochs 20 \
  --batch-size 16 \
  --target-height 96 \
  --max-width 2048 \
  --device cuda
```

Outputs:

```text
outputs/checkpoints/crnn_ctc_h96_w2048_fold0_best.pt
outputs/predictions/crnn_ctc_h96_w2048_fold0_valid_best.csv
```

## 5. Create submission

```bash
python scripts/predict_crnn_ctc.py \
  --run-name crnn_ctc_h96_w2048_fold0 \
  --checkpoint outputs/checkpoints/crnn_ctc_h96_w2048_fold0_best.pt \
  --device cuda
```

Outputs:

```text
outputs/predictions/crnn_ctc_h96_w2048_fold0_test.csv
outputs/submissions/crnn_ctc_h96_w2048_fold0_submission.csv
```

## TrOCR experiments

TrOCR uses the same manifests from step 2.

### Small default

```bash
python scripts/train_trocr.py \
  --run-name trocr_small_default_fold0 \
  --model-name microsoft/trocr-small-handwritten \
  --preprocess-mode default \
  --fold 0 \
  --epochs 10 \
  --batch-size 4 \
  --lr 5e-5 \
  --device cuda
```

### Small aspect-ratio-aware

```bash
python scripts/train_trocr.py \
  --run-name trocr_small_aspect_h384_w1536_fold0 \
  --model-name microsoft/trocr-small-handwritten \
  --preprocess-mode aspect \
  --target-height 384 \
  --canvas-width 1536 \
  --fold 0 \
  --epochs 10 \
  --batch-size 2 \
  --lr 3e-5 \
  --device cuda
```

### Base with partial freezing

```bash
python scripts/train_trocr.py \
  --run-name trocr_base_default_freeze_enc8_fold0 \
  --model-name microsoft/trocr-base-handwritten \
  --preprocess-mode default \
  --freeze-encoder-layers 8 \
  --fold 0 \
  --epochs 8 \
  --batch-size 2 \
  --lr 2e-5 \
  --device cuda
```

### Large

```bash
python scripts/train_trocr.py \
  --run-name trocr_large_default_fold0 \
  --model-name microsoft/trocr-large-handwritten \
  --preprocess-mode default \
  --fold 0 \
  --epochs 5 \
  --batch-size 1 \
  --lr 1e-5 \
  --device cuda
```

### TrOCR submission

```bash
python scripts/predict_trocr.py \
  --run-name trocr_small_default_fold0 \
  --checkpoint-dir outputs/checkpoints/trocr_small_default_fold0 \
  --device cuda
```

## ResNet-CTC experiment

This is a stronger CTC visual encoder than the first custom CRNN baseline, while
keeping the same character vocabulary, folds, CTC loss, and greedy decoder.

```bash
python scripts/train_resnet_ctc.py \
  --run-name resnet_ctc_h96_w2048_fold0 \
  --fold 0 \
  --epochs 30 \
  --batch-size 12 \
  --target-height 96 \
  --max-width 2048 \
  --lr 7e-4 \
  --device cuda
```

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w2048_fold0 \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w2048_fold0_best.pt \
  --device cuda
```

## Ensemble CRNN + TrOCR

First create test predictions from each trained model. Then combine them.

```bash
python scripts/ensemble_predictions.py \
  --run-name ensemble_crnn_trocr_fold0 \
  --prediction crnn:outputs/predictions/crnn_ctc_h96_w2048_fold0_test.csv \
  --prediction trocr:outputs/predictions/trocr_small_default_fold0_test.csv \
  --priority trocr crnn \
  --sample-submission /path/to/SampleSubmission.csv
```

For validation predictions, use the `*_valid_best.csv` files instead. The
script will also save WER/CER/score when a `reference` column exists.

## Kaggle path example

Kaggle input paths usually look like this:

```bash
python scripts/make_folds.py \
  --train-csv /kaggle/input/<dataset-name>/Train.csv \
  --test-csv /kaggle/input/<dataset-name>/Test.csv \
  --sample-submission /kaggle/input/<dataset-name>/SampleSubmission.csv \
  --image-dir /kaggle/input/<dataset-name>/images
```

Use the Kaggle notebook file browser to confirm the exact folder name.

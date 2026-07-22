# Cloud Training Workflow

Use this repository as the source of truth. Edit code locally, push to GitHub,
then pull or reclone it inside Kaggle, Colab, RunPod, or Lambda.

## 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

On Kaggle/Colab, PyTorch is usually preinstalled. If dependency resolution is
slow, install only the missing packages.

For Kaggle, avoid reinstalling PyTorch because it can break CUDA package
compatibility with the preinstalled runtime. Use:

```bash
python -m pip uninstall -y torchaudio
python -m pip install -r requirements-kaggle.txt
```

`torchaudio` is not used by this OCR project. Removing it avoids a known
`transformers` import failure when PyTorch and TorchAudio are compiled with
different CUDA versions.

## Multi-GPU

Training scripts automatically use `torch.nn.DataParallel` when `--device cuda`
and more than one CUDA GPU is available, such as Kaggle `GPU T4 x2`.

Supported training scripts:

```text
scripts/train_crnn_ctc.py
scripts/train_resnet_ctc.py
scripts/train_convnext_ctc.py
scripts/train_trocr.py
```

Disable it only if debugging:

```bash
--no-data-parallel
```

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

## CTC decoding experiments

These experiments reuse an existing CTC checkpoint. They do not retrain the
visual model. Run them against validation first, compare WER/CER, then create a
test submission only for the best decoding setup.

### Beam search without language model

Validation prediction:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam10_valid \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --test-manifest data/metadata/train_manifest.csv \
  --fold 0 \
  --predictions-output outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam10_valid.csv \
  --decoder beam \
  --beam-size 10 \
  --candidates-top-k 5 \
  --no-submission \
  --device cuda
```

Evaluate it:

```bash
python scripts/diagnose_evaluator.py \
  --predictions outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam10_valid.csv
```

Test submission:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam10 \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --decoder beam \
  --beam-size 10 \
  --candidates-top-k 5 \
  --device cuda
```

### Character n-gram LM

Train a lightweight character LM from `Train.csv`:

```bash
python scripts/train_char_lm.py \
  --train-csv /path/to/Train.csv \
  --char-vocab data/metadata/char_vocab.json \
  --order 4 \
  --add-k 0.5 \
  --output outputs/language_models/char_ngram_order4.json
```

Validation prediction:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_valid \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --test-manifest data/metadata/train_manifest.csv \
  --fold 0 \
  --predictions-output outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_valid.csv \
  --decoder beam_lm \
  --beam-size 25 \
  --lm-path outputs/language_models/char_ngram_order4.json \
  --lm-weight 0.05 \
  --candidates-top-k 5 \
  --no-submission \
  --device cuda
```

Test submission:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005 \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --decoder beam_lm \
  --beam-size 25 \
  --lm-path outputs/language_models/char_ngram_order4.json \
  --lm-weight 0.05 \
  --candidates-top-k 5 \
  --device cuda
```

### Beam search + LM + conservative reranking

Use this only after validating that plain beam or beam+LM is competitive.

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_rerank_valid \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --test-manifest data/metadata/train_manifest.csv \
  --fold 0 \
  --predictions-output outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_rerank_valid.csv \
  --decoder beam_lm_rerank \
  --beam-size 25 \
  --lm-path outputs/language_models/char_ngram_order4.json \
  --lm-weight 0.05 \
  --candidates-top-k 5 \
  --rerank-repeated-whitespace-penalty 0.2 \
  --rerank-repeated-punctuation-penalty 0.2 \
  --rerank-edge-space-penalty 0.2 \
  --no-submission \
  --device cuda
```

## ConvNeXt-CTC experiment

This uses a `timm` ConvNeXt-tiny encoder with ImageNet initialization, followed
by the same BiLSTM, CTC loss, character vocabulary, and greedy CTC decoder.

```bash
python scripts/train_convnext_ctc.py \
  --run-name convnext_ctc_h96_w2048_fold0 \
  --fold 0 \
  --epochs 20 \
  --batch-size 8 \
  --target-height 96 \
  --max-width 2048 \
  --lr 3e-4 \
  --device cuda
```

```bash
python scripts/predict_ctc.py \
  --run-name convnext_ctc_h96_w2048_fold0 \
  --checkpoint outputs/checkpoints/convnext_ctc_h96_w2048_fold0_best.pt \
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

## Diagnostic forensics

Before running many full training jobs, use the diagnostics layer to audit the
pipeline. See `docs/diagnostics.md` for the full sequence.

```bash
python scripts/diagnose_charset.py
```

```bash
python scripts/diagnose_ctc_alignment.py \
  --target-height 96 \
  --max-width 2048 \
  --time-downsample-factor 4
```

```bash
python scripts/make_overfit_subset.py \
  --n-samples 64 \
  --require-chars "^*"
```

After a validation run finishes, evaluate the saved validation predictions:

```bash
python scripts/diagnose_evaluator.py \
  --predictions outputs/predictions/crnn_ctc_h96_w2048_fold0_valid_best.csv
```

```bash
python scripts/diagnose_error_groups.py \
  --predictions outputs/predictions/crnn_ctc_h96_w2048_fold0_valid_best.csv
```

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

## PyLaia HTR experiment

PyLaia uses an external HTR workflow. See `docs/pylaia_training.md` for the
full sequence:

```text
prepare_pylaia_dataset.py -> validate_pylaia_dataset.py -> download_pylaia_model.py -> train_pylaia.py -> predict_pylaia.py -> convert_pylaia_predictions.py
```

Note that PyLaia 1.1.2 supports Python 3.9 and 3.10, so use a Python 3.10 cloud
environment for this branch of experiments.

## Kraken HTR experiment

Kraken uses another external historical HTR workflow. See
`docs/kraken_training.md` for the full sequence:

```text
prepare_kraken_dataset.py -> download_kraken_model.py -> train_kraken.py -> test_kraken.py -> predict_kraken.py -> convert_kraken_predictions.py
```

## PaddleOCR recognition experiment

PaddleOCR/SVTR/PP-OCR is a lower-priority external OCR recognizer workflow. See
`docs/paddleocr_training.md` for the full sequence:

```text
prepare_paddleocr_dataset.py -> make_paddleocr_config.py -> train_paddleocr_rec.py -> eval_paddleocr_rec.py -> predict_paddleocr_rec.py -> convert_paddleocr_predictions.py
```

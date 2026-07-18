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

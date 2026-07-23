# TrOCR Workflow

TrOCR diperlakukan sebagai pipeline terpisah dari CTC, tetapi tetap memakai fondasi repo yang sama:

- `data/metadata/train_manifest.csv`
- `data/metadata/test_manifest.csv`
- fold-based validation
- local CER/WER
- checkpoint di `outputs/checkpoints/`
- prediction CSV di `outputs/predictions/`
- submission CSV di `outputs/submissions/`

## Supported Scenarios

Pipeline TrOCR dibuat configurable sehingga beberapa skenario bisa dijalankan dari CLI yang sama.

| Scenario | Model | Preprocessing | Notes |
| --- | --- | --- | --- |
| Small default | `microsoft/trocr-small-handwritten` | default HF processor | Baseline pretrained tercepat. |
| Small aspect-ratio-aware | `microsoft/trocr-small-handwritten` | fixed height + canvas width | Mengurangi risiko line image gepeng karena square resize. |
| Base partial freezing | `microsoft/trocr-base-handwritten` | default | Menekan overfit/VRAM dengan freeze sebagian encoder. |
| Large default | `microsoft/trocr-large-handwritten` | default | Mahal; hanya untuk eksperimen pembanding. |

## Dependencies

TrOCR membutuhkan dependency Hugging Face:

```text
transformers
accelerate
sentencepiece
evaluate
```

Di Kaggle/Colab, jika muncul konflik `torchaudio`, uninstall `torchaudio` sebelum install requirements Kaggle:

```bash
!pip uninstall -y torchaudio
!pip install -r /kaggle/working/barbados-historic-handwriting-ocr/requirements-kaggle.txt
```

## Training Examples

Small default:

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_trocr.py \
  --run-name trocr_small_default_fold0 \
  --model-name microsoft/trocr-small-handwritten \
  --preprocess-mode default \
  --fold 0 \
  --epochs 10 \
  --batch-size 4 \
  --lr 5e-5 \
  --device cuda
```

Small aspect-ratio-aware:

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_trocr.py \
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

Base with partial freezing:

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_trocr.py \
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

Large default:

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/train_trocr.py \
  --run-name trocr_large_default_fold0 \
  --model-name microsoft/trocr-large-handwritten \
  --preprocess-mode default \
  --fold 0 \
  --epochs 5 \
  --batch-size 1 \
  --lr 1e-5 \
  --device cuda
```

## Prediction Example

```bash
!python /kaggle/working/barbados-historic-handwriting-ocr/scripts/predict_trocr.py \
  --run-name trocr_small_default_fold0 \
  --checkpoint-dir /kaggle/working/barbados-historic-handwriting-ocr/outputs/checkpoints/trocr_small_default_fold0 \
  --sample-submission "/kaggle/input/datasets/furqanalghifari/historic-handwriting-comp/raw/road-barbados-historic-handwriting-challenge/SampleSubmission.csv" \
  --device cuda
```

## Lessons Learned

- TrOCR is task-relevant, but current public submissions underperformed the strongest ResNet-CTC branch.
- Default square preprocessing is risky for very wide handwriting lines.
- Aspect-ratio-aware preprocessing is conceptually better, but not guaranteed to outperform without careful tuning.
- TrOCR can hallucinate plausible words; CTC tends to be more literal.
- Keep TrOCR as a complementary model candidate for ensemble or future transfer experiments, not the current primary route.

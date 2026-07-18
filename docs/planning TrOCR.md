Planning saya: kita buat **jalur TrOCR sebagai pipeline terpisah** dari CRNN-CTC, tapi tetap mengikuti pola repo yang sama: manifest, fold, CLI, checkpoint, valid predictions, submission.

**Target Arsitektur**
Kita dukung 4 mode lewat satu kode yang configurable:

1. `trocr-small-default`
   - checkpoint: `microsoft/trocr-small-handwritten`
   - preprocessing default processor Hugging Face
   - paling cepat dibuat dan jadi baseline pretrained

2. `trocr-small-aspect`
   - checkpoint sama
   - preprocessing preserve aspect ratio + pad canvas
   - untuk menghindari line image dipaksa persegi

3. `trocr-base-freeze`
   - checkpoint: `microsoft/trocr-base-handwritten`
   - partial freezing encoder/decoder
   - targetnya akurasi lebih tinggi dengan risiko overfit/VRAM lebih terkontrol

4. `trocr-large`
   - checkpoint: `microsoft/trocr-large-handwritten`
   - prioritas terakhir, compute berat

**Struktur Kode Yang Akan Dibuat**
- `src/data/trocr_dataset.py`
  Dataset untuk Hugging Face processor, label tokenization, collate.

- `src/training/trocr_trainer.py`
  Training loop atau wrapper Hugging Face `Seq2SeqTrainer`.

- `src/inference/trocr_predictor.py`
  Load checkpoint, generate text, valid/test prediction, submission.

- `scripts/train_trocr.py`
  CLI training:
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

- `scripts/predict_trocr.py`
  CLI inference/submission.

**Dependency Tambahan**
Perlu tambah:
```text
transformers
accelerate
sentencepiece
```

Kemungkinan juga:
```text
evaluate
```

Tapi karena metric WER/CER kita sudah punya sendiri, `evaluate` tidak wajib.

**Desain Penting**
- Tetap pakai `train_manifest.csv` dan `test_manifest.csv`.
- Validasi tetap pakai WER/CER lokal kita.
- Output tetap:
  - `outputs/checkpoints/{run_name}/`
  - `outputs/predictions/{run_name}_valid_best.csv`
  - `outputs/predictions/{run_name}_test.csv`
  - `outputs/submissions/{run_name}_submission.csv`

**Urutan Implementasi**
1. Tambah dependency Hugging Face.
2. Buat TrOCR dataset + collator.
3. Buat train script untuk `trocr-small-default`.
4. Buat predict script.
5. Setelah default stabil, tambah `aspect-ratio-aware`.
6. Tambah freeze options untuk Base.
7. Large tidak perlu kode khusus, cukup ganti `--model-name`.

Jadi kita tidak perlu membuat 4 pipeline berbeda. Kita buat **satu pipeline TrOCR configurable** yang bisa menjalankan Small default, Small aspect-ratio-aware, Base freeze, dan Large hanya dari parameter CLI.
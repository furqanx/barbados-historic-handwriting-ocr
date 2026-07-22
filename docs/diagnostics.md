# Diagnostic Forensics

Lapisan ini dipakai untuk membuktikan bahwa pipeline OCR/HTR sehat sebelum
melakukan tuning model besar-besaran.

Semua output default disimpan ke:

```text
reports/diagnostics/
```

## 1. Charset dan tokenizer round-trip

```bash
python scripts/diagnose_charset.py \
  --train-csv data/raw/road-barbados-historic-handwriting-challenge/Train.csv \
  --vocab data/metadata/char_vocab.json
```

Output penting:

- `character_frequency.csv`
- `tokenizer_roundtrip.csv`
- `rare_character_lines.csv`

Tujuan:

- memastikan semua karakter target ada di vocabulary;
- memastikan `decode(encode(text)) == text`;
- menemukan line yang mengandung karakter langka.

## 2. CTC alignment audit

```bash
python scripts/diagnose_ctc_alignment.py \
  --train-manifest data/metadata/train_manifest.csv \
  --target-height 96 \
  --max-width 2048 \
  --time-downsample-factor 4
```

Output penting:

- `ctc_alignment_audit.csv`
- `ctc_alignment_summary.json`

Tujuan:

- memastikan `encoder_time_steps >= target_length + adjacent_repeat_count`;
- melihat berapa line yang terpotong oleh `max-width`;
- melihat `pixels_per_char` untuk line panjang.

## 3. TrOCR token/generation audit

```bash
python scripts/diagnose_trocr_targets.py \
  --train-manifest data/metadata/train_manifest.csv \
  --model-name microsoft/trocr-small-handwritten \
  --preprocess-mode aspect \
  --target-height 384 \
  --canvas-width 1536 \
  --max-label-length 192 \
  --max-generation-length 192
```

Output penting:

- `trocr_target_audit.csv`
- `trocr_target_summary.json`

Tujuan:

- memastikan target token tidak terpotong saat training;
- memastikan generation length cukup saat inference;
- mengecek risiko clipping pada aspect-aware canvas.

## 4. Small-subset overfit manifest

```bash
python scripts/make_overfit_subset.py \
  --train-manifest data/metadata/train_manifest.csv \
  --n-samples 64 \
  --require-chars "^*"
```

Output default:

```text
data/metadata/overfit_manifest_64_rarechars.csv
```

Manifest ini menggandakan sampel yang sama menjadi split train dan valid:

- `fold != 0` untuk train;
- `fold == 0` untuk valid.

Contoh training CRNN untuk tes hafalan kecil:

```bash
python scripts/train_crnn_ctc.py \
  --train-manifest data/metadata/overfit_manifest_64_rarechars.csv \
  --run-name overfit_crnn_64 \
  --fold 0 \
  --epochs 80 \
  --batch-size 8 \
  --target-height 96 \
  --max-width 2048 \
  --device cuda
```

Target diagnostik:

- training/validation CER mendekati 0;
- WER mendekati 0;
- simbol langka seperti `^` dan `*` tidak hilang.

Jika gagal menghafal 32-100 gambar, jangan lanjut tuning full dataset dulu.

## 5. Evaluasi prediksi validation

```bash
python scripts/diagnose_evaluator.py \
  --truth data/metadata/train_manifest.csv \
  --predictions outputs/predictions/crnn_ctc_h96_w2048_fold0_valid_best.csv
```

Secara default script mengevaluasi hanya ID yang ada di prediction CSV, sehingga
cocok untuk file validasi satu fold. Tambahkan `--require-all-truth` bila ingin
memaksa prediksi mencakup seluruh truth file.

Output penting:

- `evaluation_summary.json`
- `row_errors.csv`
- `character_confusions.csv`

Tujuan:

- memakai satu evaluator untuk semua workflow;
- melihat row terburuk;
- melihat insertion/deletion/substitution;
- melihat confusion karakter.

## 6. Error analysis per kelompok

```bash
python scripts/diagnose_error_groups.py \
  --manifest data/metadata/train_manifest.csv \
  --predictions outputs/predictions/crnn_ctc_h96_w2048_fold0_valid_best.csv \
  --rare-chars "^*#|\\"
```

Output penting:

- `grouped_row_errors.csv`
- `grouped_error_summary.csv`

Tujuan:

- membandingkan CER/WER pada short/medium/long line;
- mengecek line dengan simbol langka;
- mengecek punctuation-heavy atau digit-heavy line;
- melihat apakah model cenderung deletion atau insertion.

## Urutan yang direkomendasikan

1. `diagnose_charset.py`
2. `diagnose_ctc_alignment.py`
3. `diagnose_trocr_targets.py` jika memakai TrOCR
4. `make_overfit_subset.py`
5. Train model pada overfit manifest
6. `diagnose_evaluator.py`
7. `diagnose_error_groups.py`

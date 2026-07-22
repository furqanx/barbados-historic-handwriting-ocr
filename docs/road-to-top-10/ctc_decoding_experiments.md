# CTC Decoding Experiments

Dokumen ini mencatat konfigurasi penting untuk fase CTC decoding. Fokus fase
ini adalah memakai checkpoint ResNet-CTC terbaik yang sudah ada, lalu mengganti
cara decoding dari logits menjadi teks.

Artinya: visual model tidak dilatih ulang.

## Skenario

Tiga skenario utama:

1. `CTC beam search tanpa LM`
2. `CTC beam search + char LM ringan`
3. `CTC reranking`

Semua eksperimen harus diuji di validation dulu. Jangan langsung submit ke test.

## 1. Beam Search Tanpa LM

Parameter utama:

- `--beam-size`
- `--top-tokens-per-step`
- `--length-bonus`

Nilai awal yang layak dicoba:

- `beam-size`: `5`, `10`, `25`
- `top-tokens-per-step`: `8` atau `10`
- `length-bonus`: `0.0` dulu

Command awal:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam10_valid \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --test-manifest data/metadata/train_manifest.csv \
  --fold 0 \
  --predictions-output outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam10_valid.csv \
  --decoder beam \
  --beam-size 10 \
  --top-tokens-per-step 10 \
  --candidates-top-k 5 \
  --no-submission \
  --device cuda
```

## 2. Beam Search + Character LM

Parameter utama:

- `--beam-size`
- `--lm-weight`
- `--length-bonus`
- `--top-tokens-per-step`

Nilai awal yang layak dicoba:

- `beam-size`: `10` atau `25`
- `lm-weight`: `0.02`, `0.05`, `0.10`
- `length-bonus`: `0.0`, lalu `0.01` jika output terlalu pendek
- `top-tokens-per-step`: `10`

Latih character LM:

```bash
python scripts/train_char_lm.py \
  --train-csv data/raw/road-barbados-historic-handwriting-challenge/Train.csv \
  --char-vocab data/metadata/char_vocab.json \
  --order 4 \
  --add-k 0.5 \
  --output outputs/language_models/char_ngram_order4.json
```

Command awal:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_valid \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --test-manifest data/metadata/train_manifest.csv \
  --fold 0 \
  --predictions-output outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_valid.csv \
  --decoder beam_lm \
  --beam-size 25 \
  --top-tokens-per-step 10 \
  --lm-path outputs/language_models/char_ngram_order4.json \
  --lm-weight 0.05 \
  --candidates-top-k 5 \
  --no-submission \
  --device cuda
```

Catatan penting:

- LM harus ringan.
- Bobot LM yang terlalu besar bisa menghapus simbol langka seperti `^`, `*`, `#`.
- Kalau validation CER memburuk, turunkan `lm-weight`.

## 3. Reranking

Parameter utama:

- `--rerank-repeated-whitespace-penalty`
- `--rerank-repeated-punctuation-penalty`
- `--rerank-edge-space-penalty`
- `--rerank-min-chars`
- `--rerank-short-text-penalty`

Nilai awal yang layak dicoba:

- `rerank-repeated-whitespace-penalty`: `0.1` atau `0.2`
- `rerank-repeated-punctuation-penalty`: `0.1` atau `0.2`
- `rerank-edge-space-penalty`: `0.1` atau `0.2`
- `rerank-min-chars`: jangan dipakai dulu
- `rerank-short-text-penalty`: jangan dipakai dulu

Command awal:

```bash
python scripts/predict_ctc.py \
  --run-name resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_rerank_valid \
  --checkpoint outputs/checkpoints/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_best.pt \
  --test-manifest data/metadata/train_manifest.csv \
  --fold 0 \
  --predictions-output outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_beam25_lm005_rerank_valid.csv \
  --decoder beam_lm_rerank \
  --beam-size 25 \
  --top-tokens-per-step 10 \
  --lm-path outputs/language_models/char_ngram_order4.json \
  --lm-weight 0.05 \
  --candidates-top-k 5 \
  --rerank-repeated-whitespace-penalty 0.2 \
  --rerank-repeated-punctuation-penalty 0.2 \
  --rerank-edge-space-penalty 0.2 \
  --no-submission \
  --device cuda
```

## Urutan Eksperimen Hemat

Urutan yang disarankan:

1. `beam_size=10`, tanpa LM
2. `beam_size=25`, tanpa LM
3. `beam_size=25`, `lm_weight=0.02`
4. `beam_size=25`, `lm_weight=0.05`
5. `beam_size=25`, `lm_weight=0.10`
6. konfigurasi terbaik dari atas + rerank ringan

Jangan tweak terlalu banyak di awal. Tujuan pertama adalah mencari apakah beam
search dan LM memberikan sinyal membaik di validation.

## Evaluasi

Setelah membuat validation prediction, jalankan:

```bash
python scripts/diagnose_evaluator.py \
  --predictions outputs/predictions/<prediction_valid_file>.csv
```

Yang dipantau:

- `score`
- `cer`
- `wer`
- deletion/insertion/substitution
- apakah simbol langka makin sering hilang
- apakah output makin terlalu pendek

Submission test baru dibuat setelah validation membaik.

# Hyperparameter Notes

Dokumen ini menyimpan catatan hyperparameter yang pernah dipakai atau disarankan. Gunakan sebagai referensi awal, bukan resep final.

## CRNN-CTC Baseline

CRNN-CTC dari random initialization cukup sensitif terhadap:

- target image height;
- max width;
- batch size;
- learning rate;
- jumlah epoch;
- decoder yang dipakai saat inference.

Pada epoch awal, WER/CER bisa terlihat sangat buruk (`~1.0`) dan itu normal untuk CTC. Jangan menyimpulkan run gagal dari epoch 1-3 saja; tunggu loss turun cukup jauh dan cek valid prediction.

Eksperimen awal yang masuk akal:

```bash
--run-name crnn_ctc_h96_w2048_bs16_lr1e3_fold0 \
--epochs 30 \
--batch-size 16 \
--target-height 96 \
--max-width 2048 \
--lr 1e-3
```

```bash
--run-name crnn_ctc_h64_w1536_bs24_lr1e3_fold0 \
--epochs 30 \
--batch-size 24 \
--target-height 64 \
--max-width 1536 \
--lr 1e-3
```

```bash
--run-name crnn_ctc_h96_w2048_bs16_lr3e4_fold0 \
--epochs 30 \
--batch-size 16 \
--target-height 96 \
--max-width 2048 \
--lr 3e-4
```

## ResNet-CTC

ResNet-CTC sejauh ini menjadi keluarga model CTC terbaik.

Konfigurasi utama:

```bash
--target-height 96 \
--max-width 2048 \
--batch-size 12 \
--lr 7e-4 \
--epochs 30
```

Eksperimen width lebih besar:

```bash
--target-height 96 \
--max-width 3072 \
--batch-size 8 \
--lr 7e-4 \
--epochs 30
```

Hasil public score menunjukkan width `3072` hanya naik tipis dari baseline. Pelajarannya: memperbesar canvas membantu sedikit, tetapi bukan jalur lompatan besar menuju `0.90`.

## Fold Training

Fold training benar-benar melatih model ulang pada split berbeda.

Strategi hemat compute:

1. Train fold 0 sebagai baseline.
2. Train fold 1 dengan konfigurasi terbaik.
3. Bandingkan single fold dan selective ensemble.
4. Lanjut fold 2-4 hanya jika fold 1 memberi sinyal positif.

Hasil saat ini menunjukkan fold 1 menjadi single-model anchor terbaik secara public score.

## CTC Decoding

Decoding tidak melatih ulang visual model.

Yang dapat dituning:

- `beam-size`: kandidat umum `10`, `25`, `50`.
- `top-tokens-per-step`: kandidat umum `8`, `12`, `20`.
- `candidates-top-k`: kandidat umum `3`, `5`, `10`.
- `lm-weight`: kandidat awal `0.01`, `0.02`, `0.05`.
- `length-bonus`: gunakan hati-hati; validasi dulu.

Validasi awal menunjukkan `beam10` dan `beam25` sangat mirip, dan 5-fold beam majority tidak mengalahkan fold 1 anchor. Artinya decoding membantu sedikit, tetapi tidak cukup sebagai strategi utama.

## TrOCR

Starting point:

- Small default: `lr 5e-5`, batch size `4`, epoch `10`.
- Small aspect: `lr 3e-5`, batch size `2`, target height `384`, canvas width `1536`.
- Base freeze: `lr 2e-5`, batch size `2`, epoch `8`, freeze encoder layers `8`.
- Large: `lr 1e-5`, batch size `1`, epoch `5`.

TrOCR membutuhkan kontrol overfit dan preprocessing. Default square resize kurang ideal untuk line image sangat lebar.

## Practical Rule

Untuk eksperimen baru:

1. Jalankan pipeline validation lebih dulu.
2. Lakukan small-subset overfit test jika ada model/pipeline baru.
3. Gunakan satu fold dulu.
4. Submit hanya jika validasi memberi alasan jelas.
5. Catat public score dan file submission di `competition_history.md`.

## Planned Historical HTR Grid

Grid besar yang sudah disetujui:

```text
PyLaia: 5 checkpoints x 5 folds = 25 training runs
Kraken: 3 checkpoints x 5 folds = 15 training runs
Total: 40 training runs
```

PyLaia checkpoints:

- `himanis`
- `belfort`
- `norhand-v1`
- `norhand-v3`
- `iam`

Kraken checkpoints:

- `catmus-medieval`
- `mccatmus`
- `tridis`

Post-train decoding variants:

- native/default decode;
- beam search;
- beam search + character LM;
- beam search + character LM + reranking;
- validation-based post-processing if only final text is available.

# ResNet Fold 1 Artifact Review

Tanggal review: 2026-07-21

ZIP sumber:

```text
ResNet-CTC-3_training_artifacts.zip
```

## File Yang Ditempatkan

Checkpoint:

```text
outputs/checkpoints/resnet_ctc_h96_w2048_fold1_best.pt
```

Predictions:

```text
outputs/predictions/resnet_ctc_h96_w2048_fold1_valid_best.csv
outputs/predictions/resnet_ctc_h96_w2048_fold1_test.csv
```

Submission:

```text
outputs/submissions/resnet_ctc_h96_w2048_fold1_submission.csv
```

Metadata dari runtime cloud disimpan sebagai arsip, bukan mengganti metadata
lokal:

```text
data/metadata/artifacts/ResNet-CTC-3_training_artifacts/
```

## Local Validation Metrics

| Run | Fold | Rows | Score | WER | CER | Exact Match |
|---|---:|---:|---:|---:|---:|---:|
| `resnet_ctc_h96_w2048_fold0` | 0 | 820 | 0.205872 | 0.321123 | 0.090621 | 0.059756 |
| `resnet_ctc_h96_w3072_bs8_lr7e4_fold0` | 0 | 820 | 0.207150 | 0.321881 | 0.092419 | 0.059756 |
| `resnet_ctc_h96_w2048_fold1` | 1 | 820 | 0.202264 | 0.315014 | 0.089514 | 0.064634 |

Catatan:

- Fold 1 terlihat sedikit lebih baik dari fold 0 secara local validation.
- Perbandingan fold 0 vs fold 1 tidak sepenuhnya apple-to-apple karena
  validation set-nya berbeda.
- Sinyalnya tetap positif: training fold tambahan layak dipakai untuk
  ensemble/test diversity.

## Fold 1 Error Group Notes

Laporan dibuat di:

```text
reports/diagnostics/error_groups/resnet_ctc_h96_w2048_fold1_valid_best/
```

Pola error fold 1:

- `digit_heavy=True` paling berat: mean char edits sekitar `12.42`.
- `punctuation_heavy=True` juga berat: mean char edits sekitar `10.35`.
- `has_rare_symbol=True` lebih sulit daripada tanpa rare symbol.
- `prediction_shorter=True` sangat buruk: exact match `0.0`, mean char edits
  sekitar `7.43`.
- Aspect ratio rendah (`low_ar`) lebih sulit daripada `mid_ar` dan `high_ar`.

Ini menguatkan prioritas berikutnya:

- CTC beam search untuk mengurangi deletion.
- Char LM ringan dengan bobot kecil, hati-hati agar simbol langka tidak hilang.
- Post-processing harus divalidasi khusus pada punctuation dan rare symbol.

## Test Ensemble Yang Dibuat

File ensemble:

```text
outputs/predictions/ensemble_resnet_w3072_f0_w2048_f0_w2048_f1.csv
outputs/submissions/ensemble_resnet_w3072_f0_w2048_f0_w2048_f1_submission.csv
```

Komponen:

```text
resnet_w3072_f0 = outputs/predictions/resnet_ctc_h96_w3072_bs8_lr7e4_fold0_test.csv
resnet_w2048_f0 = outputs/predictions/resnet_ctc_h96_w2048_fold0_test.csv
resnet_w2048_f1 = outputs/predictions/resnet_ctc_h96_w2048_fold1_test.csv
```

Priority tie-break:

```text
resnet_w3072_f0
resnet_w2048_f1
resnet_w2048_f0
```

Agreement di test:

| Agreement Type | Rows |
|---|---:|
| all three same | 40 |
| at least two same | 153 |
| all different | 1221 |

Interpretasi:

- Mayoritas baris test memiliki tiga prediksi yang berbeda.
- Dengan hanya text-level voting, ensemble masih sangat bergantung pada model
  prioritas.
- Untuk lompatan skor besar, ensemble text saja kemungkinan tidak cukup.
  Decoding beam/LM dan confidence-aware strategy lebih penting.

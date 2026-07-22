# Diagnosis Training Artifacts

Artefak training dari tiga ZIP sudah diekstrak.

## Penempatan

- File dari `outputs/` ditempatkan ke `outputs/`.
- Metadata dari ZIP tidak menimpa metadata lokal. Metadata disimpan di:
  - `data/metadata/artifacts/CRNN-CTC_training_artifacts/`
  - `data/metadata/artifacts/ResNet_n_ConvNeXt_training_artifacts/`
  - `data/metadata/artifacts/TrOCR_training_artifacts/`

Alasan metadata dipisahkan: manifest dari cloud dapat berisi path Kaggle/Colab,
sedangkan manifest lokal harus tetap memakai path laptop.

## Output Diagnosis

Aggregate report:

```text
reports/diagnostics/model_predictions/validation_summary.csv
reports/diagnostics/model_predictions/validation_summary_enriched.csv
```

Setiap run juga memiliki:

```text
reports/diagnostics/model_predictions/<run_name>/row_errors.csv
reports/diagnostics/model_predictions/<run_name>/character_confusions.csv
reports/diagnostics/model_predictions/<run_name>/grouped_row_errors.csv
reports/diagnostics/model_predictions/<run_name>/grouped_error_summary.csv
```

## Ringkasan Validasi Fold 0

| Rank | Run | Score | WER | CER | Exact Match |
|---:|---|---:|---:|---:|---:|
| 1 | `trocr_large_default_fold0` | 0.190762 | 0.268126 | 0.113399 | 0.089024 |
| 2 | `resnet_ctc_h96_w2048_fold0` | 0.205872 | 0.321123 | 0.090621 | 0.059756 |
| 3 | `crnn_ctc_h96_w2048_bs16_lr1e3_fold0` | 0.208814 | 0.323941 | 0.093688 | 0.053659 |
| 4 | `crnn_ctc_h96_w2048_fold0` | 0.224498 | 0.347025 | 0.101971 | 0.043902 |
| 5 | `trocr_base_default_freeze_enc8_fold0` | 0.246914 | 0.331202 | 0.162626 | 0.065854 |
| 6 | `crnn_ctc_h96_w2048_bs16_lr3e4_fold0` | 0.250007 | 0.385716 | 0.114297 | 0.029268 |
| 7 | `crnn_ctc_h64_w1536_bs24_lr1e3_fold0` | 0.250610 | 0.386258 | 0.114962 | 0.019512 |
| 8 | `trocr_small_aspect_h384_w1536_fold0` | 0.277210 | 0.369134 | 0.185286 | 0.034146 |
| 9 | `trocr_small_default_fold0` | 0.401753 | 0.496694 | 0.306812 | 0.010976 |
| 10 | `convnext_ctc_h96_w2048_fold0` | 0.977535 | 1.000000 | 0.955070 | 0.000000 |

## Interpretasi

- Skor terbaik saat ini: `trocr_large_default_fold0`.
- CER terbaik saat ini: `resnet_ctc_h96_w2048_fold0`.
- `resnet_ctc_h96_w2048_fold0` dan `crnn_ctc_h96_w2048_bs16_lr1e3_fold0`
  sangat kompetitif dan jauh lebih ringan daripada TrOCR Large.
- `convnext_ctc_h96_w2048_fold0` gagal secara praktis: semua 820 validation
  prediction kosong atau hampir kosong. Ini perlu debugging training/prediction
  khusus sebelum ConvNeXt dipakai lagi.

## Rare Symbol Lines

Semua model dievaluasi pada 116 validation rows yang mengandung rare symbol.

| Run | Rare Symbol Mean Row CER |
|---|---:|
| `resnet_ctc_h96_w2048_fold0` | 0.135423 |
| `crnn_ctc_h96_w2048_bs16_lr1e3_fold0` | 0.137348 |
| `trocr_large_default_fold0` | 0.142043 |
| `crnn_ctc_h96_w2048_fold0` | 0.145099 |
| `crnn_ctc_h64_w1536_bs24_lr1e3_fold0` | 0.157757 |
| `crnn_ctc_h96_w2048_bs16_lr3e4_fold0` | 0.160792 |
| `trocr_base_default_freeze_enc8_fold0` | 0.222034 |
| `trocr_small_aspect_h384_w1536_fold0` | 0.242100 |
| `trocr_small_default_fold0` | 0.382454 |
| `convnext_ctc_h96_w2048_fold0` | 0.955856 |

Kesimpulan rare symbol: CTC, terutama ResNet/CRNN, lebih literal dan lebih kuat
untuk simbol langka dibanding TrOCR Small/Base pada hasil ini. TrOCR Large tetap
kuat secara overall.

## Prioritas Berikutnya

1. Debug `convnext_ctc_h96_w2048_fold0` karena output validation kosong.
2. Gunakan `resnet_ctc_h96_w2048_fold0` sebagai baseline CTC utama.
3. Gunakan `trocr_large_default_fold0` sebagai baseline autoregressive terbaik,
   tetapi pertimbangkan biaya GPU.
4. Lakukan error analysis detail dari:
   - `reports/diagnostics/model_predictions/trocr_large_default_fold0/`
   - `reports/diagnostics/model_predictions/resnet_ctc_h96_w2048_fold0/`
   - `reports/diagnostics/model_predictions/crnn_ctc_h96_w2048_bs16_lr1e3_fold0/`
5. Setelah model individual stabil, ensemble kandidat paling masuk akal:
   `trocr_large_default_fold0` + `resnet_ctc_h96_w2048_fold0`.


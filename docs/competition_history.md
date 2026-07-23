# Diagnosis And Experiment History

Dokumen ini adalah catatan diagnosis utama untuk kompetisi
**R.O.A.D. Barbados Historic Handwriting Challenge**. Isinya menggabungkan
riwayat artifact, validasi model, public leaderboard, dan keputusan eksperimen
yang sebelumnya tersebar di beberapa file markdown.

## Competition Context

Task kompetisi ini adalah **line-level handwritten text recognition** untuk
dokumen historis Barbados. Submission berupa CSV dengan kolom:

```text
ID,Target
```

Tantangan utama bukan hanya membaca kata, tetapi juga mempertahankan:

- kapitalisasi;
- spasi;
- tanda baca;
- angka;
- singkatan historis;
- simbol langka seperti `^`, `*`, `#`, `|`, dan `\`.

Target pribadi: meningkatkan public leaderboard score menuju **0.90+** agar
mulai kompetitif untuk area top leaderboard.

## Data And Pipeline Diagnostics

| Area | Result | Interpretation |
|---|---:|---|
| Train rows | 4,098 | Dataset relatif kecil, sehingga overfitting dan pretraining perlu diperhatikan. |
| Test rows | 1,374 | Submission size manageable. |
| Readable images | 5,472 / 5,472 | Tidak ada gambar corrupt. |
| Empty labels | 0 | Tidak ada label kosong. |
| Unique characters | 81 | Character-level vocabulary cukup kecil dan cocok untuk CTC. |
| Tokenizer round-trip failures | 0 | CTC tokenizer lossless terhadap training labels. |
| Invalid CTC alignment rows | 0 | Dengan `h96/w2048/stride4`, semua target punya cukup CTC time steps. |
| CTC width-clipped rows | 979 | Banyak line berpotensi kehilangan detail visual saat `max-width=2048`. |
| TrOCR max token count | 35 | Target length aman untuk `max_label_length=192`. |
| TrOCR aspect canvas clipped rows | 4,096 | `canvas-width=1536` terlalu kecil untuk TrOCR aspect-aware pada sebagian besar line. |

Kesimpulan awal: pipeline CTC sehat. TrOCR aspect-aware perlu perhatian khusus
pada ukuran canvas/lebar gambar.

## Training Artifacts

Artifact ZIP dari cloud training sudah diekstrak dan ditempatkan ke lokasi
kerja. File ZIP mentah disimpan rapi di:

```text
outputs/exports/training_artifacts/
```

Artifact penting:

```text
outputs/checkpoints/
outputs/predictions/
outputs/submissions/
```

Metadata dari runtime cloud tidak menimpa metadata lokal, karena manifest cloud
sering berisi path Kaggle/Colab. Metadata cloud disimpan sebagai arsip:

```text
data/metadata/artifacts/
```

## Local Validation Summary

Validation awal menggunakan fold 0, 820 rows. Metric lokal memakai:

```text
score = 0.5 * WER + 0.5 * CER
```

Semakin rendah semakin baik untuk error lokal. Public leaderboard Zindi
menampilkan score yang semakin tinggi semakin baik, sehingga angka lokal dan
public tidak dibandingkan secara langsung.

| Rank | Run | Local Score | WER | CER | Exact Match |
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

Interpretasi:

- `trocr_large_default_fold0` terbaik secara local validation score.
- `resnet_ctc_h96_w2048_fold0` terbaik secara CER lokal.
- CTC lebih literal untuk karakter dan simbol.
- `convnext_ctc_h96_w2048_fold0` gagal secara praktis dan perlu debugging
  khusus sebelum dipakai lagi.

## Rare Symbol Validation

Baris validation yang mengandung rare symbol berjumlah 116 rows.

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
kuat secara overall, tetapi public leaderboard menunjukkan ia belum robust untuk
test set.

## ResNet Fold Diagnostics

### Fold 1 Artifact

ZIP sumber:

```text
ResNet-CTC-3_training_artifacts.zip
```

File utama:

```text
outputs/checkpoints/resnet_ctc_h96_w2048_fold1_best.pt
outputs/predictions/resnet_ctc_h96_w2048_fold1_valid_best.csv
outputs/predictions/resnet_ctc_h96_w2048_fold1_test.csv
outputs/submissions/resnet_ctc_h96_w2048_fold1_submission.csv
```

Local validation:

| Run | Fold | Rows | Score | WER | CER | Exact Match |
|---|---:|---:|---:|---:|---:|---:|
| `resnet_ctc_h96_w2048_fold0` | 0 | 820 | 0.205872 | 0.321123 | 0.090621 | 0.059756 |
| `resnet_ctc_h96_w3072_bs8_lr7e4_fold0` | 0 | 820 | 0.207150 | 0.321881 | 0.092419 | 0.059756 |
| `resnet_ctc_h96_w2048_fold1` | 1 | 820 | 0.202264 | 0.315014 | 0.089514 | 0.064634 |

Catatan:

- Fold 1 sedikit lebih baik dari fold 0 secara local validation.
- Perbandingan antar fold tidak sepenuhnya apple-to-apple karena validation
  set-nya berbeda.
- Public leaderboard kemudian membuktikan fold 1 menjadi single-model anchor
  terbaik.

### Fold 1 Error Pattern

Pola error fold 1:

- `digit_heavy=True` paling berat.
- `punctuation_heavy=True` juga berat.
- `has_rare_symbol=True` lebih sulit daripada tanpa rare symbol.
- `prediction_shorter=True` sangat buruk dan mengarah ke masalah deletion.
- Aspect ratio rendah lebih sulit daripada mid/high aspect ratio.

Implikasi:

- CTC beam search bisa dicoba untuk mengurangi deletion, tetapi efek public
  ternyata kecil.
- Char LM harus dipakai hati-hati agar tidak menghapus simbol langka.
- Post-processing harus divalidasi, terutama pada punctuation dan rare symbol.

## Beam25 5-Fold Review

Submission:

```text
outputs/submissions/ensemble_resnet_ctc_5fold_beam25_majority_submission.csv
public score: 0.799222877
```

Hasil ini lebih buruk daripada fold1 anchor.

Pairwise agreement penting:

| Pair | Same rows | Different rows | Same rate |
|---|---:|---:|---:|
| beam25 5-fold majority vs `resnet_ctc_h96_w2048_fold0_submission.csv` | 992 | 382 | 0.721980 |
| beam25 5-fold majority vs `resnet_ctc_h96_w2048_fold1_submission.csv` | 170 | 1204 | 0.123726 |
| beam25 5-fold majority vs `ensemble_resnet_w3072_f0_w2048_f0_w2048_f1_submission.csv` | 129 | 1245 | 0.093886 |
| beam25 5-fold majority vs `resnet_ctc_h96_w3072_bs8_lr7e4_fold0_submission.csv` | 91 | 1283 | 0.066230 |

Interpretasi:

- Majority vote 5-fold beam25 terlalu kasar.
- Ia mengganti terlalu banyak prediksi dari anchor yang kuat.
- Ensemble teks harus selektif, bukan majority vote buta.

## Public Leaderboard History

| Submission File | Public Score | WER Weighted | CER Weighted | Note |
|---|---:|---:|---:|---|
| `selective_fold1_consensus3of4_submission.csv` | 0.803358324 | 3.513759677 | 5.525852503 | Current best public score; fold1 anchor with 35 selective replacements. |
| `selective_fold1_consensus4of4_submission.csv` | 0.802836148 | 3.523466122 | 5.538803957 | Conservative selective ensemble with 2 replacements. |
| `resnet_ctc_h96_w2048_fold1_submission.csv` | 0.802726676 | unknown | unknown | Strongest single-model public submission. |
| `ensemble_resnet_ctc_5fold_beam25_majority_submission.csv` | 0.799222877 | 3.621276119 | 5.487967896 | 5-fold beam25 majority; worse than fold1 anchor. |
| `resnet_ctc_h96_w3072_bs8_lr7e4_fold0_submission.csv` | 0.794606501 | 3.703126474 | 5.62062178 | Wider image run; only slight gain over fold0. |
| `resnet_ctc_h96_w2048_fold0_submission.csv` | 0.793236683 | 3.723841083 | 5.676359895 | Earlier ResNet baseline. |
| `trocr_large_default_fold0_submission.csv` | 0.734202099 | 3.772521247 | 11.94704672 | Strong local validation, weak public CER. |

Current best benchmark copy:

```text
outputs/submissions/best/current_best_public_0_803358324_selective_fold1_consensus3of4_submission.csv
```

## Selective Ensemble Findings

Anchor:

```text
outputs/submissions/resnet_ctc_h96_w2048_fold1_submission.csv
public score: 0.802726676
```

Selective ensemble results:

| Run | Strategy | Changed rows | Public Score |
|---|---|---:|---:|
| `selective_fold1_consensus4of4` | Replace only if all four other folds agree | 2 / 1374 | 0.802836148 |
| `selective_fold1_consensus3of4` | Replace if any three other folds agree | 35 / 1374 | 0.803358324 |
| `selective_fold1_length_guarded_consensus2of4` | Replace only if anchor length is an outlier | 0 / 1374 | not submitted |
| `selective_fold1_weighted_anchor3` | Weighted vote with anchor weight 3 | 2 / 1374 | not submitted |

Kesimpulan: selective replacement yang menjaga fold1 sebagai anchor berhasil
menaikkan public score. Kenaikannya kecil, tetapi secara strategi lebih sehat
daripada majority vote.

## Current Lessons Learned

1. **CTC adalah anchor terbaik saat ini.**
   ResNet-CTC adalah keluarga model paling reliable di public leaderboard.

2. **TrOCR belum menjadi jalur utama.**
   TrOCR Large kuat di local validation, tetapi public CER buruk.

3. **Local validation dan public leaderboard tidak sepenuhnya sejalan.**
   Keputusan eksperimen harus mempertimbangkan public feedback.

4. **Majority vote buta gagal.**
   5-fold beam25 majority menurunkan public score.

5. **Selective replacement berhasil.**
   Fold1 anchor + consensus replacement memberi current best public score.

6. **CTC kecil-kecilan mulai diminishing returns.**
   Kenaikan dari selective ensemble nyata tetapi kecil. Untuk menuju 0.90,
   kemungkinan perlu sumber sinyal baru: pretrained historical HTR,
   better data strategy, pseudo-label, atau post-processing yang lebih cerdas.

7. **ConvNeXt perlu debugging sebelum eksperimen lanjutan.**
   Validation output sebelumnya kosong/hampir kosong.

## Current Best Model Candidates

| Role | Candidate |
|---|---|
| Best public submission | `selective_fold1_consensus3of4_submission.csv` |
| Best single-model public anchor | `resnet_ctc_h96_w2048_fold1_submission.csv` |
| Best local validation score | `trocr_large_default_fold0` |
| Best local CTC CER baseline | `resnet_ctc_h96_w2048_fold0` |
| Useful diversity candidate | `crnn_ctc_h96_w2048_bs16_lr1e3_fold0` |
| Needs debugging | `convnext_ctc_h96_w2048_fold0` |

## Next Direction

CTC should not be discarded because it is the best anchor, but further small
CTC decoding/ensemble tweaks are unlikely to produce a large jump.

Recommended next direction:

1. Preserve current best benchmark.
2. Pause minor CTC variants unless they are strongly motivated.
3. Revisit historical pretrained HTR once environment issues are solved.
4. Explore data-centric improvements:
   - pseudo-labeling;
   - targeted correction rules validated on error groups;
   - stronger image preprocessing for long/low aspect lines;
   - external language priors if competition rules allow.


# Experiment History

Dokumen ini mencatat progres eksperimen untuk kompetisi
**R.O.A.D. Barbados Historic Handwriting Challenge**. Tujuannya bukan hanya
menyimpan angka, tetapi juga menjaga konteks keputusan eksperimen: model mana
yang kuat, model mana yang gagal, dan mengapa langkah berikutnya dipilih.

## Competition Context

Task kompetisi ini adalah **line-level handwritten text recognition** untuk
dokumen historis Barbados. Target submission berupa CSV berisi `ID` dan
transkripsi teks. Karena domainnya handwriting historis, tantangan utamanya
tidak hanya membaca kata, tetapi juga mempertahankan:

- kapitalisasi;
- spasi;
- tanda baca;
- angka;
- singkatan historis;
- simbol langka seperti `^`, `*`, `#`, `|`, dan `\`.

Target pribadi saat ini adalah meningkatkan public leaderboard score menuju
sekitar **0.90+** agar mulai kompetitif untuk area top leaderboard.

## Dataset Diagnostics Summary

Sebelum membandingkan model, pipeline data sudah diaudit.

| Area | Result | Interpretation |
|---|---:|---|
| Train rows | 4,098 | Dataset relatif kecil, sehingga overfitting dan pretraining perlu diperhatikan. |
| Test rows | 1,374 | Submission size manageable. |
| Readable images | 5,472 / 5,472 | Tidak ada gambar corrupt. |
| Empty labels | 0 | Tidak ada label kosong. |
| Unique characters | 81 | Character-level vocabulary cukup kecil dan cocok untuk CTC. |
| Tokenizer round-trip failures | 0 | CTC tokenizer lossless terhadap training labels. |
| Invalid CTC alignment rows | 0 | Dengan `h96/w2048/stride4`, semua target punya cukup CTC time steps. |
| CTC width-clipped rows | 979 | Ada banyak line yang detail visualnya mungkin dipadatkan oleh `max-width=2048`. |
| TrOCR max token count | 35 | Target length aman untuk `max_label_length=192`. |
| TrOCR aspect canvas clipped rows | 4,096 | `canvas-width=1536` untuk TrOCR aspect-aware terlalu kecil untuk sebagian besar line. |

Kesimpulan diagnosis awal: pipeline CTC sehat, sedangkan TrOCR aspect-aware
perlu perhatian khusus pada lebar canvas.

## Local Validation Results

Validation menggunakan fold 0, berisi 820 rows. Metric lokal menggunakan
`score = 0.5 * WER + 0.5 * CER`, semakin rendah semakin baik untuk error
lokal. Public leaderboard Zindi menampilkan score yang semakin tinggi semakin
baik, sehingga angka lokal dan public tidak dibandingkan secara langsung.

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

### Local Interpretation

`trocr_large_default_fold0` menjadi model terbaik di validation lokal secara
score gabungan, terutama karena WER-nya lebih baik daripada CTC. Namun
`resnet_ctc_h96_w2048_fold0` memiliki CER terbaik. Ini penting karena task OCR
historis sangat sensitif terhadap karakter dan simbol kecil.

CRNN-CTC masih kompetitif dan tidak jauh dari ResNet-CTC. Hal ini menunjukkan
bahwa pipeline CTC sudah bekerja dengan benar. Sebaliknya,
`convnext_ctc_h96_w2048_fold0` gagal secara praktis karena validation
predictions kosong atau hampir kosong, sehingga perlu debugging sebelum dipakai
lagi.

## Rare Symbol Validation

Baris validation yang mengandung rare symbols berjumlah 116 rows.

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

Interpretasi: CTC, terutama ResNet-CTC dan CRNN-CTC, lebih literal terhadap
simbol langka. TrOCR Large tetap kuat secara overall, tetapi pretrained
autoregressive decoder dapat memiliki language bias yang lebih besar daripada
CTC.

## Public Leaderboard Submissions

Hasil public leaderboard awal menunjukkan pola yang berbeda dari validation
lokal. Di public leaderboard, ResNet-CTC mengungguli TrOCR Large.

| Submission File | Public Score | WER Weighted | CER Weighted | Note |
|---|---:|---:|---:|---|
| `resnet_ctc_h96_w2048_fold0_submission.csv` | 0.793236683 | 3.723841083 | 5.676359895 | Best public score so far. |
| `trocr_large_default_fold0_submission.csv` | 0.734202099 | 3.772521247 | 11.94704672 | Strong local validation, weaker public CER. |
| Other early submission | 0.735052492 | 3.759843141 | 11.91161147 | Similar to TrOCR-style result. |

### Public Leaderboard Interpretation

Public results suggest that **ResNet-CTC is the current primary direction**.
Although TrOCR Large had the best local validation score, its public CER
weighted metric is much worse than ResNet-CTC. This indicates either:

- local validation does not fully match the public test distribution;
- TrOCR predictions are less robust on public test handwriting;
- TrOCR decoder is more likely to produce character-level substitutions;
- CTC is better aligned with literal transcription conventions.

For future decisions, public leaderboard signal should be weighted strongly.

## Lessons Learned

1. **CTC is the strongest practical baseline.**
   ResNet-CTC has the best public score and the best local CER.

2. **TrOCR Large is valuable but not primary yet.**
   It can help as a diversity model or ensemble member, but current public
   evidence does not support using it alone as the main submission.

3. **Local validation and public leaderboard disagree.**
   This makes grouped error analysis and additional folds important.

4. **Resolution is likely a high-value lever.**
   The CTC alignment audit is safe, but 979 training images are width-clipped
   at `max-width=2048`. Larger `max-width` or `target-height` may improve
   long lines and rare symbols.

5. **ConvNeXt needs debugging before further experiments.**
   Its validation output is effectively empty. This is likely a training,
   optimization, checkpoint, or output-length issue rather than a meaningful
   model comparison.

## Current Best Model Candidates

| Role | Candidate |
|---|---|
| Best public baseline | `resnet_ctc_h96_w2048_fold0` |
| Best local validation score | `trocr_large_default_fold0` |
| Best local CER | `resnet_ctc_h96_w2048_fold0` |
| Best rare-symbol behavior | `resnet_ctc_h96_w2048_fold0` |
| Lightweight competitive model | `crnn_ctc_h96_w2048_bs16_lr1e3_fold0` |
| Needs debugging | `convnext_ctc_h96_w2048_fold0` |

## Next Experiment Plan

The immediate objective is improving the ResNet-CTC baseline toward public
score 0.90+.

Recommended next runs:

1. `resnet_ctc_h96_w3072_bs8_lr7e4_fold0`
2. `resnet_ctc_h128_w3072_bs4_lr7e4_fold0`

Rationale:

- `w3072` reduces width clipping and may preserve long-line detail.
- `h128` may improve small marks, punctuation, and historical abbreviations.
- Batch size is reduced to control GPU memory.

After these runs:

1. submit both models;
2. compare public score against `resnet_ctc_h96_w2048_fold0`;
3. run validation diagnostics on their `*_valid_best.csv`;
4. if score improves, repeat across additional folds;
5. test simple ensemble between best ResNet-CTC and TrOCR Large.

## Portfolio Summary

This phase established a disciplined OCR experiment loop:

- audited data integrity and tokenizer round-trip behavior;
- validated CTC alignment constraints;
- compared CTC and autoregressive OCR families;
- diagnosed rare-symbol behavior;
- reconciled local validation with public leaderboard feedback;
- selected the next experiment direction based on evidence.

The main technical conclusion is that for this historical handwriting dataset,
**a character-level CTC recognizer with strong visual encoding is currently more
reliable than a larger autoregressive model for leaderboard performance**.


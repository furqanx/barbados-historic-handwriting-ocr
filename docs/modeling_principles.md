# Modeling Principles And Experiment Strategy

Dokumen ini merangkum prinsip modeling dan pelajaran teknis yang harus dijaga
sebelum mencoba pendekatan baru, sehingga eksperimen berikutnya tetap terarah
dan tidak mengulang kesalahan yang sama.

## Task Reality

Kompetisi ini bukan OCR modern biasa. Task-nya adalah **historical handwritten
line recognition** dengan target transkripsi literal.

Model harus mempertahankan:

- ejaan historis;
- kapitalisasi;
- tanda baca;
- angka;
- simbol seperti `^`, `*`, `#`, `|`, `~`, dan apostrof;
- spasi internal;
- fragmen kalimat yang tidak lengkap;
- bentuk kata yang mungkin tampak “salah” secara ejaan modern.

Sistem tidak boleh sekadar menghasilkan kalimat yang masuk akal. Output harus
sedekat mungkin dengan anotasi literal.

## Label And Normalization Policy

Normalisasi teks hanya boleh memperbaiki representasi teknis, bukan isi label.

Yang harus dipertahankan:

- uppercase/lowercase;
- ejaan historis;
- simbol khusus;
- tanda baca;
- urutan karakter;
- multiple spaces jika evaluator menghitungnya;
- bentuk asli kata meskipun tidak sesuai ejaan modern.

Yang perlu diaudit:

- Unicode NFC vs NFD;
- apostrof lurus vs apostrof melengkung;
- hyphen/en dash/em dash;
- regular space vs non-breaking space;
- tab/newline/carriage return tersembunyi;
- leading/trailing whitespace;
- jumlah spasi berurutan.

Satu fungsi normalisasi resmi harus digunakan bersama oleh:

- preprocessing label;
- tokenizer;
- training;
- decoding;
- evaluator;
- converter submission.

Submission final tetap harus mengikuti evaluator kompetisi, bukan versi
normalized yang hanya membuat skor lokal terlihat lebih bagus.

## Evaluation Contract

Sebelum membandingkan model, seluruh workflow harus dievaluasi dengan aturan
yang sama:

- split/fold yang sama;
- normalisasi teks yang sama;
- aturan whitespace yang sama;
- Unicode handling yang sama;
- implementasi CER/WER/score yang sama;
- checkpoint selection berdasarkan metrik yang sama.

Evaluator pusat harus bisa menghasilkan:

- CER;
- WER;
- competition score;
- insertion/deletion/substitution count;
- exact-match rate;
- row-level error table;
- character confusion table.

Loss bukan metrik utama untuk memilih model final. Pilih checkpoint berdasarkan
validation CER/WER/competition score, lalu gunakan public feedback sebagai sinyal
tambahan karena validation dan public leaderboard terbukti tidak selalu sejalan.

## Pipeline Sanity Checks

Sebelum hyperparameter tuning besar:

1. Audit manifest, path, image, dan label.
2. Audit charset dan tokenizer round-trip.
3. Audit target truncation.
4. Audit CTC sequence length.
5. Audit evaluator consistency.
6. Jalankan small-subset overfit test pada 32-100 gambar.

Small-subset overfit test harus mampu menghasilkan training CER/WER mendekati 0.
Jika gagal, kemungkinan masalahnya ada pada pipeline, bukan arsitektur:

- vocabulary tidak lengkap;
- simbol berubah/hilang saat preprocessing;
- target terpotong;
- blank/padding salah;
- time step CTC kurang;
- attention mask salah;
- decoder berhenti terlalu awal;
- converter output mengubah prediksi.

## Fourteen Engineering Surfaces

Kompetisi ini perlu dilihat sebagai proses engineering end-to-end, bukan hanya
hyperparameter tuning.

| Area | What To Review |
|---|---|
| Data integrity | File corrupt, label kosong, ID duplikat, image duplikat, line terpotong, mismatch gambar-label. |
| Split and validation | Distribusi panjang teks, simbol langka, handwriting style, image quality, aspect ratio, leakage. |
| Text normalization | Unicode, whitespace, apostrof, dash, hidden chars, simbol langka. |
| Charset and tokenization | CTC blank, unknown char, encode/decode lossless, TrOCR tokenization symbol. |
| Image preprocessing | RGB/grayscale, contrast, denoising, sharpening, binarization, margin crop. |
| Image geometry | `target-height`, `max-width`, `canvas-width`, aspect ratio, clipping, pixels per character. |
| Augmentation | Brightness, contrast, blur ringan, noise ringan, affine kecil, ink fading. |
| Sampling | Rare-symbol sampling, long-line sampling, quality-aware sampling, hard-example sampling. |
| Architecture | Horizontal stride, feature-map height, sequence length, encoder strength, dropout. |
| Objective and alignment | CTC input length, target length, repeated chars, label masking, EOS behavior. |
| Optimization | LR, scheduler, batch size, gradient clipping, weight decay, freezing. |
| Transfer learning | Checkpoint domain similarity, charset, period, language, transcription convention. |
| Decoding/post-processing | Greedy, beam, char LM, generation length, technical cleanup only. |
| Error analysis/ensemble | Grouped CER/WER, character recall, disagreement, selective ensemble. |

Priority:

- **P0:** data integrity, text normalization, charset/tokenization, geometry,
  alignment, evaluator consistency.
- **P1:** rare-symbol sampling, augmentation, optimization, transfer learning,
  decoding.
- **P2:** advanced architecture, multi-fold confirmation, ensemble,
  post-processing.

## Image Geometry

Line images are long and thin. Median aspect ratio is high, so horizontal
geometry matters.

Principles:

- resize to fixed height;
- preserve aspect ratio;
- pad to the right;
- avoid square resizing for CTC-style line recognizers;
- avoid aggressive horizontal compression;
- avoid cropping internal gaps;
- track clipping rate and pixels per character.

Important parameters:

- CTC: `target-height`, `max-width`, horizontal stride, input lengths,
  width bucketing.
- TrOCR: `target-height`, `canvas-width`, preserve-ratio preprocessing,
  visual token count, generation max length.

Height 96 or 128 is often safer than 64 for:

- punctuation;
- `^`;
- `*`;
- tiny strokes;
- ambiguous historical letter shapes.

## CTC Principles

CTC is the most literal family in the current workspace.

Audit:

- blank index;
- target length;
- input length;
- repeated adjacent characters;
- repeated spaces;
- invalid alignment;
- blank dominance;
- horizontal stride;
- `zero_infinity`.

For CTC, effective time step requirement is stricter than target length:

```text
required_steps = target_length + adjacent_repeated_character_count
```

Do not rely on `zero_infinity=True` to hide invalid samples. Log invalid rows.

CTC loss and CER do not necessarily move together. Loss can improve while greedy
decoding stagnates. Compare:

1. greedy decoding;
2. beam search without LM;
3. beam search with lightweight character LM.

Character LM must be weak and carefully validated. If it is too strong, it may
modernize spelling or remove symbols.

## TrOCR Principles

TrOCR has a stronger language prior than CTC. This can help WER, but can hurt
literal transcription and rare symbols.

Audit:

- target token length after tokenization;
- `max_length` / `max_new_tokens`;
- EOS behavior;
- padding labels;
- decoder start token;
- `clean_up_tokenization_spaces=False`;
- special token cleanup;
- stability for `^`, `*`, multiple spaces, and historical punctuation.

Avoid aggressive NLP generation constraints:

- high repetition penalty;
- `no_repeat_ngram_size`;
- aggressive length penalty;
- early stopping that cuts line endings.

Start with:

- greedy generation;
- beam size 2-5;
- neutral length penalty;
- no spell correction.

## Historical HTR Transfer

PyLaia and Kraken should be treated as **domain transfer experiments**, not just
extra architectures.

Their value depends on:

- handwriting period similarity;
- script/alphabet similarity;
- language similarity;
- transcription convention;
- charset compatibility;
- Unicode normalization.

Before full fine-tuning:

1. Run zero-shot or small validation prediction.
2. Inspect whether output structure is plausible.
3. Audit charset and symbol handling.
4. Confirm environment stability.

Previous practical lesson:

- PyLaia requires a compatible Python/runtime stack.
- Kraken can be memory-sensitive depending on model and segmentation workflow.

Current approved HTR training grid:

- PyLaia checkpoints: `himanis`, `belfort`, `norhand-v1`, `norhand-v3`, `iam`.
- Kraken checkpoints: `catmus-medieval`, `mccatmus`, `tridis`.
- Folds: 0, 1, 2, 3, 4.
- Full grid: 40 training runs.

Post-train decoding should not stop at greedy/default decode. Evaluate native
beam search, character-LM decoding, and reranking where the external workflow
supports it. If PyLaia/Kraken expose only final text, run validation-based
post-processing and ensemble diagnostics instead of pretending we can reuse the
PyTorch CTC logits decoder.

## PaddleOCR / Scene Text Workflow

PaddleOCR/SVTR/PP-OCR is a backup/diversity workflow, not the main HTR path.

Reason:

- many checkpoints target printed/scene/short-word OCR;
- max text length and aspect ratio assumptions may not match long historical
  lines;
- it can still be useful as diversity if carefully adapted.

## Error Analysis

Do not judge experiments by one global CER/WER number only.

Group rows by:

- short/medium/long line;
- rare-symbol line;
- punctuation-heavy line;
- digit-heavy line;
- multiple-space line;
- high/low aspect ratio;
- low-quality image;
- dense vs sparse writing;
- prediction shorter/longer than target.

Classify error type:

- deletion;
- insertion;
- substitution;
- spacing;
- uppercase/lowercase;
- punctuation;
- symbol missing;
- truncated ending;
- repeated-character error.

Examples of actionable patterns:

| Dominant Error | Likely Action |
|---|---|
| Ending often missing | Check generation max length, EOS, image clipping, input compression. |
| Repeated chars missing in CTC | Check time steps, stride, blank behavior, decoding. |
| `^` / `*` missing | Check rarity, resolution, augmentation, LM strength. |
| Spacing wrong | Check label normalization and visual gap handling. |
| Training poor | Pipeline/alignment/vocabulary/learning rate issue. |
| Validation poor but training good | Overfitting/split/domain shift. |

## Experiment Method

Experiments should change one major variable at a time.

For CTC:

1. `target-height`;
2. `max-width`;
3. horizontal stride / sequence length;
4. encoder pretraining;
5. learning rate;
6. effective batch size;
7. RNN hidden size;
8. augmentation;
9. dropout/weight decay;
10. decoding.

For TrOCR:

1. preprocessing default vs aspect-aware;
2. `target-height`;
3. `canvas-width`;
4. target truncation and generation length;
5. learning rate;
6. freezing;
7. effective batch size;
8. augmentation;
9. beam configuration.

Do not change `target-height`, `max-width`, LR, batch size, and augmentation all
at once. If score changes, the cause becomes unclear.

## Active Workflow Inventory

| # | Workflow | Train | Predict | Role |
|---:|---|---|---|---|
| 1 | CRNN + BiLSTM + CTC | `scripts/train_crnn_ctc.py` | `scripts/predict_crnn_ctc.py` | Lightweight diagnostic baseline. |
| 2 | ResNet-style CNN + BiLSTM + CTC | `scripts/train_resnet_ctc.py` | `scripts/predict_ctc.py` | Current strongest public anchor. |
| 3 | ConvNeXt-tiny + BiLSTM + CTC | `scripts/train_convnext_ctc.py` | `scripts/predict_ctc.py` | Modern CTC visual encoder, needs debugging. |
| 4 | TrOCR-Small default | `scripts/train_trocr.py` | `scripts/predict_trocr.py` | IAM pretrained autoregressive baseline. |
| 5 | TrOCR-Small aspect-aware | `scripts/train_trocr.py` | `scripts/predict_trocr.py` | More suitable TrOCR variant for long line images. |
| 6 | TrOCR-Base partial freezing | `scripts/train_trocr.py` | `scripts/predict_trocr.py` | Data-small pretrained experiment. |
| 7 | TrOCR-Large | `scripts/train_trocr.py` | `scripts/predict_trocr.py` | Scaling experiment, expensive. |
| 8 | CSV ensemble | none | `scripts/ensemble_predictions.py`, `scripts/selective_ensemble_predictions.py` | Final selection/ensemble layer. |
| 9 | PyLaia HTR | `scripts/train_pylaia.py` | `scripts/predict_pylaia.py` | Historical HTR transfer workflow. |
| 10 | Kraken HTR | `scripts/train_kraken.py` | `scripts/predict_kraken.py` | Historical manuscript transfer workflow. |
| 11 | PaddleOCR recognizer | `scripts/train_paddleocr_rec.py` | `scripts/predict_paddleocr_rec.py` | General OCR backup/diversity workflow. |

Supporting scripts:

- PyLaia: `prepare_pylaia_dataset.py`, `download_pylaia_model.py`,
  `audit_pylaia_symbols.py`, `validate_pylaia_dataset.py`,
  `convert_pylaia_predictions.py`.
- Kraken: `prepare_kraken_dataset.py`, `download_kraken_model.py`,
  `audit_kraken_text.py`, `test_kraken.py`, `convert_kraken_predictions.py`.
- PaddleOCR: `prepare_paddleocr_dataset.py`, `audit_paddleocr_text.py`,
  `make_paddleocr_config.py`, `eval_paddleocr_rec.py`,
  `convert_paddleocr_predictions.py`.

## Recommended Experiment Phases

### Phase A: Pipeline Validation

1. Samakan evaluator.
2. Audit Unicode/whitespace.
3. Audit charset/tokenizer.
4. Audit target truncation.
5. Audit CTC alignment.
6. Run small-subset overfit.

### Phase B: Baseline Stabilization

1. Stabilkan ResNet-CTC.
2. Debug ConvNeXt-CTC before trusting it.
3. Stabilkan one TrOCR aspect-aware baseline.
4. Tune geometry before optimizer details.

### Phase C: Domain Transfer

1. Pilih one PyLaia checkpoint paling dekat.
2. Pilih one Kraken checkpoint paling dekat.
3. Audit charset/convention.
4. Fine-tune only after runtime is stable.

### Phase D: Confirmation

1. Train best setup on additional folds.
2. Track mean/variance.
3. Preserve out-of-fold prediction if possible.

### Phase E: Ensemble

1. Use strong individual models.
2. Prefer complementary error profiles.
3. Validate ensemble/rerank strategy.
4. Avoid blind majority vote.

## Current Strategic Takeaway

The project does not lack model families. It already has CTC, TrOCR, PyLaia,
Kraken, PaddleOCR, and ensemble tooling.

The highest-value work is likely:

- data and label correctness;
- visual geometry;
- charset and rare-symbol handling;
- alignment;
- decoding;
- error analysis;
- domain transfer from historical HTR;
- selective ensemble based on evidence.

CTC remains the strongest anchor, but minor CTC tweaks have diminishing returns.
The next major score jump likely requires a new signal source: better domain
transfer, better data strategy, or smarter validated post-processing.

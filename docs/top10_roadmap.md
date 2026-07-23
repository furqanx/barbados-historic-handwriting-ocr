# Roadmap To Top 10

Dokumen ini merangkum strategi menuju target top leaderboard untuk kompetisi
**R.O.A.D. Barbados Historic Handwriting Challenge**. Fokusnya adalah membedakan
jalur eksperimen yang masih layak dikejar dari jalur yang sudah memberi sinyal
lemah.

## Current Best Benchmark

Current best public submission:

```text
outputs/submissions/best/current_best_public_0_803358324_selective_fold1_consensus3of4_submission.csv
```

Leaderboard result:

| Submission | Public Score | WER Weighted | CER Weighted | Note |
|---|---:|---:|---:|---|
| `selective_fold1_consensus3of4_submission.csv` | 0.803358324 | 3.513759677 | 5.525852503 | Current best; fold1 anchor with 35 selective replacements. |
| `selective_fold1_consensus4of4_submission.csv` | 0.802836148 | 3.523466122 | 5.538803957 | Conservative selective ensemble with 2 replacements. |
| `resnet_ctc_h96_w2048_fold1_submission.csv` | 0.802726676 | unknown | unknown | Best single-model public anchor. |
| `ensemble_resnet_ctc_5fold_beam25_majority_submission.csv` | 0.799222877 | 3.621276119 | 5.487967896 | Blind majority vote; worse than anchor. |

Current lesson: **fold1 ResNet-CTC is the best anchor; selective replacement can
improve it slightly; blind majority vote can hurt.**

## What Has Been Tried

### ResNet-CTC Fold Training

The strongest family so far is:

```text
ResNet-style CNN + BiLSTM + CTC
target-height=96
max-width=2048
```

Fold1 became the best single-model public anchor.

### Wider ResNet-CTC

The `h96/w3072` experiment gave only a slight gain over fold0, but did not
surpass fold1.

```text
resnet_ctc_h96_w3072_bs8_lr7e4_fold0_submission.csv
public score: 0.794606501
```

Conclusion: width helps a little, but not enough by itself.

### Beam Search Without LM

Validation comparison across five folds:

| Decoder | Mean Local Score | Mean WER | Mean CER |
|---|---:|---:|---:|
| `beam25` | 0.198304 | 0.309197 | 0.087410 |
| `beam10` | 0.198377 | 0.309305 | 0.087450 |

`beam25` was slightly better locally, but public submission did not beat the
fold1 anchor when used through 5-fold majority.

Conclusion: beam search alone is not a major score lever.

### Beam Search + Character LM

Initial LM run with large beam was too slow:

```text
beam-size=25
top-tokens-per-step=20
```

Reason:

- long line images;
- many CTC time steps;
- Python-level beam/LM scoring;
- large beam and token expansion.

If revisited, use a lightweight setting first:

```text
beam-size=5 or 10
top-tokens-per-step=6 or 8
candidates-top-k=3
lm-weight=0.02
```

Conclusion: LM is not discarded, but it should be tested only on one fold/checkpoint
with a small configuration before expanding.

### 5-Fold Majority Vote

Blind majority vote failed:

```text
ensemble_resnet_ctc_5fold_beam25_majority_submission.csv
public score: 0.799222877
```

It changed too many rows away from the fold1 anchor.

Conclusion: majority vote is too blunt for this task.

### Selective Ensemble

Tooling:

```text
scripts/selective_ensemble_predictions.py
scripts/run_ctc_crossfold_predictions.py
```

Strategies implemented:

- `anchor`
- `consensus`
- `length_guarded_consensus`
- `weighted_vote`

Public results:

| Run | Changed Rows | Public Score | Decision |
|---|---:|---:|---|
| `selective_fold1_consensus4of4` | 2 / 1374 | 0.802836148 | Good, validated conservative replacement. |
| `selective_fold1_consensus3of4` | 35 / 1374 | 0.803358324 | Current best. |
| `selective_fold1_length_guarded_consensus2of4` | 0 / 1374 | not submitted | No effect. |
| `selective_fold1_weighted_anchor3` | 2 / 1374 | not submitted | Similar to 4-of-4. |

Conclusion: selective replacement is currently the best cheap improvement layer.

## CTC Status

CTC remains the strongest anchor, but minor CTC tweaks are now showing
**diminishing returns**.

Keep CTC for:

- anchor predictions;
- literal character handling;
- rare symbol robustness;
- diversity against autoregressive/HTR models;
- selective ensemble baseline.

Pause for now:

- more blind beam search submissions;
- more majority-vote-only submissions;
- small CTC decoding tweaks without strong validation reason.

Possible future CTC work only if cheap:

1. Lightweight LM on fold1 only.
2. Confidence/candidate-aware reranking if logits/candidates are saved.
3. More selective replacement variants around current best.

## Next High-Leverage Directions

To approach 0.90, we likely need a new signal source beyond current CTC variants.

### 1. Historical Pretrained HTR

Candidate families:

- PyLaia;
- Kraken CATMuS/McCATMuS/TRIDIS.

Goal:

```text
not just another architecture,
but transfer from historical handwriting domain.
```

Requirements before serious run:

- compatible runtime;
- charset audit;
- transcription convention audit;
- zero-shot or small validation prediction;
- memory-safe prediction/training flow.

Previous lesson:

- PyLaia version compatibility matters.
- Kraken can be killed by memory/runtime constraints if workflow is not tuned.

Planned full HTR grid:

| Family | Checkpoints | Folds | Training Count |
|---|---|---:|---:|
| PyLaia | `himanis`, `belfort`, `norhand-v1`, `norhand-v3`, `iam` | 5 | 25 |
| Kraken | `catmus-medieval`, `mccatmus`, `tridis` | 5 | 15 |
| Total | 8 checkpoints | 5 | 40 |

The user-approved broad plan is to cover all PyLaia and Kraken checkpoints
across all 5 folds. Execution can still be staged for safety, but the target
experiment inventory is 40 fine-tuning runs.

Post-train decoding variants to evaluate where supported:

- native/default CTC decode;
- beam search without language model;
- beam search with character-level language model;
- beam search with character-level language model plus conservative reranking;
- validation-based post-processing if the engine only exposes final text.

Important constraint: our internal PyTorch CTC decoder can be applied directly
only when we can access timestep logits in our expected format. For PyLaia and
Kraken, prefer native decoder/LM options first; use local post-processing when
only final text is available.

### 2. Data-Centric Improvements

Potentially high value:

- rare-symbol sampling;
- long-line sampling;
- hard-example sampling;
- label anomaly review;
- pseudo-labeling with high-confidence predictions;
- targeted augmentation for faint/small marks;
- geometry tuning for low-resolution or clipped rows.

Do not apply global text correction blindly. Historical spelling must remain
literal.

### 3. Post-Processing With Validation

Allowed style:

- technical cleanup;
- validated whitespace/punctuation correction;
- correction of systematic OCR substitutions only when proven on validation.

Avoid:

- modern spell correction;
- grammar correction;
- dictionary replacement;
- collapsing multiple spaces unless evaluator behavior proves it is safe.

Rule must improve validation before being considered for test submission.

### 4. Stronger Complementary Models

Promising complementary model roles:

| Role | Candidate |
|---|---|
| Literal anchor | ResNet-CTC |
| Contextual recognizer | TrOCR-Small aspect-aware or TrOCR-Large if fixed |
| Historical specialist | PyLaia/Kraken |
| Backup diversity | PaddleOCR/SVTR only if adapted well |

Ensemble should be selective and evidence-based, not simple majority vote.

## Short-Term Action Plan

1. Preserve current best benchmark.
2. Stop broad CTC-only exploration for now.
3. Keep selective ensemble tooling available.
4. Move next serious effort to one of:
   - historical pretrained HTR;
   - data-centric/pseudo-label strategy;
   - validated post-processing;
   - improved complementary model for ensemble.

## Submission Discipline

Before submitting:

1. Know which file is the anchor.
2. Know exactly how many rows changed from current best.
3. Inspect changed rows.
4. Prefer small, motivated changes.
5. Record score immediately in `docs/competition_history.md`.

Current “blessed” submission stays in:

```text
outputs/submissions/best/
```

# Project Documentation

Dokumentasi ini dipisahkan berdasarkan fungsi supaya mudah dipakai saat kompetisi berjalan.

## Recommended Reading Order

1. `eda_summary.md`
   Ringkasan struktur data, karakter target, distribusi image, kualitas image, dan implikasi preprocessing.

2. `modeling_principles.md`
   Prinsip modeling, sanity check pipeline, strategi validasi, dan pelajaran teknis yang harus dijaga.

3. `model_catalog.md`
   Katalog pendekatan model yang pernah dipertimbangkan: CTC, TrOCR, historical HTR, PaddleOCR, dan Donut.

4. `hyperparameter_notes.md`
   Catatan konfigurasi CRNN, ResNet-CTC, CTC decoding, fold training, dan TrOCR.

5. `cloud_training.md`
   Runbook utama untuk menjalankan training/inference di Kaggle, Colab, RunPod, atau runtime cloud lain.

6. `competition_history.md`
   Riwayat eksperimen, diagnosis, public score, benchmark terbaik, dan pelajaran dari submission.

7. `top10_roadmap.md`
   Strategi menuju target kompetitif, termasuk eksperimen yang sudah tidak terlalu menjanjikan dan arah berikutnya.

## Workflow-Specific Docs

- `trocr_workflow.md`: desain pipeline TrOCR dan variasi yang didukung.
- `pylaia_workflow.md`: workflow PyLaia/Teklia HTR.
- `kraken_workflow.md`: workflow Kraken historical HTR.
- `paddleocr_workflow.md`: workflow PaddleOCR recognizer.

## Diagnostics

- `diagnostics.md`: command forensik pipeline, tokenizer, CTC alignment, evaluator, dan error grouping.

## Current Benchmark

Benchmark publik terbaik yang dicatat saat ini:

- File: `outputs/submissions/best/current_best_public_0_803358324_selective_fold1_consensus3of4_submission.csv`
- Public score: `0.803358324`
- WER weighted: `3.513759677`
- CER weighted: `5.525852503`

Dokumen ini menjadi peta navigasi. Detail angka, alasan, dan konteks eksperimen disimpan di file tematik masing-masing.

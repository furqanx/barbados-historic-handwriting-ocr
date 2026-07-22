Bisa. Berikut versi rapi dari kesimpulan dua file `wisdom`.

**Kesimpulan Utama**
Kita tidak kekurangan model. Kita sudah punya banyak jalur: CRNN/ResNet/ConvNeXt CTC, TrOCR, PyLaia, Kraken, PaddleOCR, dan ensemble. Sekarang prioritasnya bukan menambah arsitektur, tetapi memastikan **pipeline, evaluasi, preprocessing, alignment, decoding, dan error analysis** benar.

**Prioritas Eksperimen**
1. **Validasi Pipeline**
   - Samakan evaluator CER/WER/score untuk semua model.
   - Audit Unicode, karakter langka, tokenizer, dan normalisasi.
   - Pastikan target tidak terpotong.
   - Pastikan submission tetap memakai format asli kompetisi.

2. **Small-Subset Overfit Test**
   - Latih model pada 32–100 gambar.
   - Matikan augmentasi.
   - Targetnya training CER/WER mendekati 0.
   - Jika gagal menghafal subset kecil, jangan lanjut tuning besar-besaran.

3. **Audit CTC**
   - Cek `encoder_time_steps >= target_length + adjacent_repeat_count`.
   - Pastikan `input_lengths` sesuai width asli setelah resize, bukan width padding.
   - Jangan mengandalkan `zero_infinity=True` untuk menyembunyikan sampel invalid.

4. **Audit TrOCR**
   - Cek panjang target setelah tokenization.
   - Pastikan `max_length` / `max_new_tokens` cukup.
   - Hindari decoding constraint NLP seperti `no_repeat_ngram_size` atau repetition penalty agresif.
   - Bandingkan greedy vs beam kecil.

5. **Fokus Geometri Input**
   - Dataset berisi line image panjang, jadi aspect ratio sangat penting.
   - Tuning paling bernilai:
     - `target-height`
     - `max-width` / `canvas-width`
     - preserve aspect ratio
     - pixels per character
   - Simbol kecil seperti `^`, `*`, tanda baca bisa hilang jika resolusi terlalu rendah.

6. **Error Analysis**
   - Jangan hanya lihat satu angka CER/WER.
   - Analisis error berdasarkan kelompok:
     - short/medium/long line
     - rare-symbol line
     - high aspect ratio
     - low-quality image
     - dense text
     - punctuation-heavy line
   - Kategorikan error: deletion, insertion, substitution, spacing, uppercase/lowercase, simbol hilang.

7. **Hyperparameter Tuning**
   - Dilakukan setelah pipeline terbukti sehat.
   - Ubah satu variabel per eksperimen.
   - Untuk CTC: mulai dari `target-height`, `max-width`, stride, lalu LR.
   - Untuk TrOCR: mulai dari preprocessing, `target-height`, `canvas-width`, generation length, lalu LR/freezing.

8. **PyLaia dan Kraken**
   - Gunakan sebagai eksperimen **domain transfer**, bukan sekadar model tambahan.
   - Pilih checkpoint berdasarkan kemiripan tulisan, periode dokumen, bahasa, charset, dan transcription convention.

9. **Ensemble**
   - Dilakukan di akhir.
   - Ensemble bernilai jika model punya error yang saling melengkapi.
   - CTC biasanya lebih literal, TrOCR lebih language-aware, Kraken/PyLaia bisa kuat jika domain dekat.

**Fokus Model**
Jalur aktif yang paling masuk akal:

| Jalur | Model |
|---|---|
| CTC baseline | ResNet-CTC |
| CTC modern | ConvNeXt-CTC |
| Autoregressive | TrOCR-Small aspect-aware |
| Historical transfer | PyLaia atau Kraken |

Model lain diposisikan sebagai pembanding/cadangan, bukan fokus tuning utama.
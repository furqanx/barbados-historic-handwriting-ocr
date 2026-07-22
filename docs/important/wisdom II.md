Ya. Dengan workflow yang sudah Anda miliki, **menambah model baru hampir pasti memberikan return yang lebih kecil** dibanding memperbaiki pipeline eksperimen yang ada.

Anda sudah mencakup tiga keluarga besar recognizer:

1. **CTC-based recognizer**
   CRNN, ResNet-CTC, ConvNeXt-CTC, PyLaia, Kraken.

2. **Autoregressive vision-language recognizer**
   TrOCR-Small, Base, dan Large.

3. **OCR recognizer alternatif**
   PaddleOCR/SVTR/PP-OCR.

Jadi masalah Anda sekarang bukan kekurangan variasi arsitektur. Masalahnya adalah menentukan **mengapa model yang sudah memadai belum menghasilkan CER, WER, dan competition score yang optimal**.

# Posisi workflow Anda saat ini

| Kelompok                 | Peran utama                                                         |
| ------------------------ | ------------------------------------------------------------------- |
| CRNN + BiLSTM + CTC      | Baseline diagnostik yang sederhana dan mudah dianalisis             |
| ResNet-CTC               | Baseline CTC utama dengan kapasitas lebih kuat                      |
| ConvNeXt-CTC             | Menguji apakah visual encoder yang lebih modern memberi peningkatan |
| TrOCR-Small default      | Baseline pretrained autoregressive                                  |
| TrOCR-Small aspect-aware | Kandidat TrOCR utama untuk line image panjang                       |
| TrOCR-Base               | Menguji keuntungan kapasitas dan pretraining yang lebih besar       |
| TrOCR-Large              | Eksperimen scaling, bukan prioritas utama                           |
| PyLaia                   | Transfer learning HTR line-level yang dekat dengan task             |
| Kraken                   | Transfer learning historical HTR yang dekat dengan domain           |
| PaddleOCR                | Kandidat diversifikasi atau cadangan                                |
| Ensemble                 | Tahap akhir setelah model individual stabil                         |

Ini sudah lebih dari cukup untuk kompetisi berukuran 4.100 training images. Yang perlu dilakukan sekarang adalah **menyempitkan ruang eksperimen**, bukan memperluasnya.

---

# Urutan pekerjaan yang semestinya dilakukan

## Tahap 1: Tetapkan satu kontrak evaluasi yang benar

Sebelum membandingkan model, pastikan seluruh workflow dievaluasi dengan aturan yang sama.

Semua model harus menggunakan:

* validation split atau fold yang sama;
* normalisasi teks yang sama;
* aturan spasi yang sama;
* aturan Unicode yang sama;
* implementasi CER dan WER yang sama;
* formula `Score` yang sama dengan kompetisi;
* checkpoint selection berdasarkan metrik yang sama.

Ini sangat penting karena PyLaia, Kraken, TrOCR, dan pipeline CTC dapat memiliki perilaku default berbeda terhadap:

* case sensitivity;
* Unicode NFC/NFD;
* trailing spaces;
* multiple spaces;
* tanda baca;
* special tokens;
* karakter tidak dikenal;
* newline atau carriage return.

Misalnya, prediksi visualnya mungkin benar:

```text
example^word*
```

tetapi salah satu converter menghasilkan:

```text
example ^ word *
```

atau:

```text
exampleword
```

Perbedaan seperti ini langsung meningkatkan CER dan WER, walaupun model sebenarnya mengenali simbol tersebut.

### Hasil yang harus tersedia

Buat satu evaluator pusat yang menerima pasangan:

```text
ground_truth
prediction
```

Kemudian menghasilkan:

* raw CER;
* raw WER;
* normalized CER, hanya untuk analisis;
* normalized WER, hanya untuk analisis;
* insertion count;
* deletion count;
* substitution count;
* score kompetisi.

**Submission tetap harus mengikuti evaluator kompetisi**, bukan versi normalized yang membuat skor terlihat lebih bagus.

---

## Tahap 2: Buktikan bahwa pipeline mampu belajar

Sebelum hyperparameter tuning, lakukan **small-subset overfitting test** untuk setiap keluarga model.

Anda tidak perlu melakukannya pada seluruh sebelas workflow. Cukup:

* CRNN atau ResNet-CTC;
* TrOCR-Small aspect-ratio-aware;
* PyLaia atau Kraken setelah workflow aktif.

Gunakan sekitar 32–100 sampel dan matikan augmentasi.

Model harus mampu memperoleh training CER yang sangat rendah, idealnya mendekati nol. Tujuan eksperimen ini bukan mendapatkan model kompetisi, melainkan menjawab:

> Apakah pipeline secara teknis mampu merepresentasikan target secara sempurna?

Bila tidak mampu menghafal subset kecil, kemungkinan masalahnya adalah:

* vocabulary tidak lengkap;
* simbol tidak dapat di-encode atau decode;
* target terpotong;
* konfigurasi CTC invalid;
* time step terlalu pendek;
* image resizing menghilangkan detail;
* padding atau attention mask salah;
* decoding berhenti terlalu awal;
* converter output mengubah prediksi;
* evaluasi CER/WER salah.

Dalam kondisi ini, mengganti learning rate atau backbone tidak menyelesaikan akar masalah.

---

# Tahap 3: Audit karakter dan simbol

Karena Anda menekankan karakter seperti `^`, `*`, dan tanda lainnya, ini harus menjadi jalur evaluasi khusus.

## 3.1 Statistik frekuensi karakter

Hitung frekuensi setiap karakter dalam train dan validation:

| Karakter | Jumlah kemunculan | Jumlah line | Persentase line |
| -------- | ----------------: | ----------: | --------------: |
| `a`      |               ... |         ... |             ... |
| `^`      |               ... |         ... |             ... |
| `*`      |               ... |         ... |             ... |
| `'`      |               ... |         ... |             ... |

Kemunculan total saja belum cukup. Misalnya `*` muncul 100 kali, tetapi seluruhnya berada pada lima line yang sama. Secara efektif keberagaman visualnya tetap kecil.

## 3.2 Ukur CER per karakter

CER global tidak memberi tahu apakah simbol khusus menjadi sumber masalah.

Anda perlu mengetahui:

* recall karakter `^`;
* recall karakter `*`;
* substitution paling umum;
* karakter yang paling sering dihapus;
* karakter yang paling sering ditambahkan;
* confusion pair.

Contoh:

| Ground truth | Sering diprediksi sebagai |
| ------------ | ------------------------- |
| `^`          | hilang, `'`, `v`          |
| `*`          | hilang, `.`, `x`          |
| `l`          | `1`, `i`                  |
| `rn`         | `m`                       |
| `c`          | `e`                       |

Dari sini baru dapat ditentukan apakah masalahnya:

* resolusi;
* data langka;
* bentuk tulisan ambigu;
* decoder language bias;
* anotasi.

## 3.3 Oversampling terarah

Jika simbol khusus sangat jarang, pertimbangkan:

* oversampling line yang mengandung simbol tersebut;
* rarity-aware sampler;
* memastikan setiap epoch melihat seluruh line langka;
* augmentasi ringan khusus line tersebut.

Jangan langsung menerapkan oversampling ekstrem. Bila terlalu agresif, model dapat meningkatkan recall simbol tetapi menambah insertion error.

---

# Tahap 4: Optimalkan geometri input

Untuk dataset Anda, ini kemungkinan lebih penting daripada pergantian backbone.

Aspect ratio median 16,8 berarti informasi karakter tersebar pada dimensi horizontal yang sangat panjang. Dua kegagalan umum adalah:

1. **Gambar terlalu dipadatkan** sehingga karakter dan tanda kecil hilang.
2. **Gambar terlalu banyak dipotong** karena `max-width` atau `canvas-width` tidak cukup.

## Untuk seluruh workflow

Analisis distribusi:

* original width;
* original height;
* resized width;
* persentase image yang terkena clipping;
* persentase image yang terlalu banyak di-downscale;
* target length terhadap resized width;
* pixels per character.

Gunakan indikator sederhana:

[
\text{pixels per character}
===========================

\frac{\text{resized width}}{\text{target character length}}
]

Jika sebuah line memiliki resized width 768 dan target 120 karakter:

[
768/120 = 6.4
]

Enam pixel horizontal per karakter mungkin terlalu rendah untuk membedakan tanda kecil dan tulisan historis yang rapat.

## Parameter yang harus diprioritaskan

Untuk CTC:

* `target-height`;
* `max-width`;
* horizontal encoder stride;
* actual input length setelah padding;
* width bucketing.

Untuk TrOCR:

* `target-height`;
* `canvas-width`;
* preserve-ratio logic;
* attention mask;
* jumlah patch visual;
* target token length;
* generation maximum length.

Tuning `target-height` dari 64 ke 96 atau 128 sering lebih bermakna daripada mengganti optimizer, terutama untuk simbol kecil.

---

# Tahap 5: Pisahkan masalah training dan decoding

Nilai loss, CER, dan WER tidak selalu bergerak bersama.

## Pada CTC

Anda dapat mengalami:

* training loss turun;
* validation loss turun;
* CER stagnan;
* WER tetap tinggi.

Ini dapat berarti probabilitas alignment membaik, tetapi hasil decoding belum berubah.

Bandingkan secara terkontrol:

1. greedy decoding;
2. CTC beam search tanpa language model;
3. beam search dengan character-level LM ringan.

Untuk task dengan simbol khusus, language model perlu sangat hati-hati. Language model yang terlalu kuat dapat:

* mengubah kata tidak umum menjadi kata umum;
* menghapus `^` atau `*`;
* memperbaiki ejaan yang sebenarnya harus ditranskripsikan secara literal.

Karena itu, **CER kemungkinan lebih cocok dijadikan metrik utama saat menilai decoder**, kemudian periksa apakah WER ikut membaik.

## Pada TrOCR

Bandingkan:

1. greedy generation;
2. beam size kecil, misalnya 2–5;
3. variasi length penalty yang konservatif;
4. tanpa repetition penalty;
5. tanpa `no_repeat_ngram_size`.

TrOCR memiliki language prior yang lebih kuat daripada CTC. Ini dapat membantu WER, tetapi bisa merugikan simbol dan kata historis yang tidak umum.

Periksa tiga jenis prediksi:

```text
Input terlihat:     ab^cd*
Ground truth:       ab^cd*
CTC prediction:     ab^cd
TrOCR prediction:   abcd
```

Jika kedua model menghapus simbol, masalah mungkin berada pada resolusi atau scarcity.

Jika hanya TrOCR yang menghapus simbol, kemungkinan decoder language prior terlalu kuat.

Jika hanya CTC yang gagal, kemungkinan alignment, time step, atau class rarity.

---

# Tahap 6: Lakukan error analysis berdasarkan kelompok

Jangan hanya mengurutkan eksperimen berdasarkan satu angka CER.

Bagi validation set menjadi kelompok:

| Kelompok             | Contoh kriteria                         |
| -------------------- | --------------------------------------- |
| Short line           | target < 40 karakter                    |
| Medium line          | 40–80 karakter                          |
| Long line            | >80 karakter                            |
| Extreme line         | mendekati 120 karakter                  |
| Rare-symbol line     | mengandung `^`, `*`, atau simbol langka |
| Low-quality line     | blur, noise, faded ink                  |
| Dense line           | karakter sangat rapat                   |
| Sparse line          | banyak whitespace                       |
| High aspect ratio    | lebar ekstrem                           |
| Common writing style | gaya mayoritas                          |
| Rare writing style   | writer atau style langka                |

Lalu hitung CER dan WER per kelompok.

Contoh hasil:

| Model              | Short CER | Long CER | Symbol CER | Overall CER |
| ------------------ | --------: | -------: | ---------: | ----------: |
| ResNet-CTC         |       5,1 |     14,3 |       29,8 |         8,7 |
| ConvNeXt-CTC       |       4,8 |     12,5 |       28,9 |         8,0 |
| TrOCR-Small aspect |       4,2 |     10,8 |       34,7 |         7,4 |

Dari sini terlihat bahwa TrOCR terbaik secara overall, tetapi paling buruk pada simbol. Ini informasi yang penting untuk:

* memilih model;
* menentukan preprocessing;
* melakukan ensemble;
* membuat sampler;
* mengubah decoding.

Tanpa error segmentation, Anda hanya mengetahui model A lebih baik dari model B, tetapi tidak mengetahui alasannya.

---

# Tahap 7: Baru lakukan hyperparameter tuning

Hyperparameter tuning tetap penting, tetapi harus bersifat **terarah**, bukan menggonta-ganti konfigurasi secara acak.

## Urutan hyperparameter untuk CTC

1. `target-height`;
2. `max-width`;
3. horizontal stride dan sequence length;
4. pretrained versus non-pretrained encoder;
5. learning rate;
6. batch size atau effective batch size;
7. RNN hidden size;
8. augmentation;
9. dropout dan weight decay;
10. decoder configuration.

Jangan mulai dari `rnn-hidden-size` atau `base-channels` bila gambar masih kehilangan detail akibat resize.

## Urutan hyperparameter untuk TrOCR

1. preprocessing default versus aspect-aware;
2. `target-height`;
3. `canvas-width`;
4. target truncation dan generation length;
5. learning rate;
6. encoder freezing;
7. decoder freezing;
8. effective batch size;
9. augmentation;
10. beam configuration.

## Metode eksperimen

Gunakan eksperimen satu variabel:

```text
Experiment A:
target-height = 64
semua parameter lain tetap

Experiment B:
target-height = 96
semua parameter lain tetap

Experiment C:
target-height = 128
semua parameter lain tetap
```

Setelah tinggi terbaik ditemukan, baru uji `max-width`.

Jangan melakukan eksperimen seperti:

```text
target-height berubah
max-width berubah
learning rate berubah
batch size berubah
augmentation berubah
```

Jika hasilnya membaik, Anda tidak tahu penyebabnya. Jika memburuk, Anda juga tidak tahu apa yang harus dibatalkan.

---

# Tahap 8: Gunakan PyLaia dan Kraken sebagai eksperimen transfer, bukan sekadar model tambahan

PyLaia dan Kraken sebaiknya tidak diperlakukan sebagai “arsitektur nomor berikutnya”. Nilai utamanya adalah **domain pretraining**.

Pertanyaan eksperimennya bukan:

> Apakah PyLaia lebih baik daripada ConvNeXt?

Tetapi:

> Apakah pretraining pada historical handwriting yang serupa memberikan representasi karakter lebih baik daripada ImageNet atau IAM handwriting?

Karena itu, checkpoint harus dipilih berdasarkan:

* kemiripan aksara;
* periode historis;
* bahasa;
* gaya tulisan;
* aturan transkripsi;
* cakupan charset;
* normalisasi Unicode.

Sebelum fine-tuning, lakukan prediction zero-shot atau minimal-adaptation pada beberapa sampel. Periksa:

* apakah model mengenali struktur kata;
* apakah simbol dipertahankan;
* apakah output sangat bias terhadap bahasa pretraining;
* apakah charset kompatibel.

Jika CATMuS atau TRIDIS sangat dekat dengan dataset Anda, Kraken bisa memberi lompatan besar. Namun jika domainnya jauh, model tersebut mungkin kalah dari ResNet-CTC custom yang dilatih secara bersih.

---

# Tahap 9: Ensemble hanya setelah model individual matang

Workflow ensemble Anda berguna, tetapi ensemble bukan obat untuk model yang belum dipahami.

Ensemble paling bernilai bila model memiliki error yang berbeda, misalnya:

* CTC lebih literal terhadap karakter;
* TrOCR lebih baik pada struktur kata;
* Kraken lebih baik pada bentuk tulisan historis;
* PyLaia lebih stabil pada line panjang.

Jangan hanya menggabungkan model dengan skor tertinggi. Ukur juga disagreement.

Contoh:

| Model A      | Model B      | Potensi ensemble                       |
| ------------ | ------------ | -------------------------------------- |
| ResNet-CTC   | ConvNeXt-CTC | Mungkin rendah karena error mirip      |
| ConvNeXt-CTC | TrOCR        | Tinggi karena decoding berbeda         |
| TrOCR        | Kraken       | Tinggi jika domain pretraining berbeda |
| PyLaia       | Kraken       | Bergantung checkpoint dan charset      |

Karena ensemble script Anda bekerja pada level CSV, kemungkinan metode saat ini adalah voting atau pemilihan prediksi utuh. Ini tetap dapat berguna, tetapi tidak sekuat confidence-aware sequence ensemble. Karena Anda tidak ingin menambah alur baru, gunakan ensemble CSV sebagai seleksi berdasarkan validation:

* majority vote bila tiga model menghasilkan string sama;
* pilih model utama pada disagreement;
* pilih model khusus untuk kelompok tertentu bila tersedia aturan yang dapat divalidasi.

Jangan membuat aturan berdasarkan test set.

---

# Workflow eksperimen yang saya rekomendasikan

## Fase A: Validasi pipeline

1. Samakan evaluator.
2. Audit Unicode dan charset.
3. Audit target truncation.
4. Audit CTC sequence length.
5. Overfit 32–100 sampel.
6. Verifikasi simbol `^`, `*`, dan karakter langka.

## Fase B: Stabilkan baseline

1. Pilih satu model CTC utama: ResNet-CTC atau ConvNeXt-CTC.
2. Pilih satu model autoregressive utama: TrOCR-Small aspect-aware.
3. Gunakan satu fold tetap.
4. Tuning geometri input.
5. Tuning learning rate.
6. Tuning augmentation.
7. Tuning decoder.

## Fase C: Domain transfer

1. Pilih satu PyLaia checkpoint yang paling dekat.
2. Pilih satu Kraken checkpoint yang paling dekat.
3. Audit charset dan transcription convention.
4. Fine-tune dengan evaluasi yang sama.
5. Bandingkan error per kelompok, bukan hanya skor overall.

## Fase D: Konfirmasi

1. Jalankan model terbaik pada fold tambahan.
2. Hitung mean dan variasi CER/WER.
3. Pastikan peningkatan bukan kebetulan split.
4. Simpan out-of-fold prediction bila memungkinkan.

## Fase E: Ensemble

1. Pilih model dengan skor individual kuat.
2. Pilih model dengan error saling melengkapi.
3. Validasi strategi ensemble pada validation atau OOF.
4. Buat submission final.

---

# Model mana yang sebaiknya menjadi fokus?

Saya akan mempersempit eksperimen aktif menjadi empat jalur:

| Jalur               | Model utama              | Tujuan                                  |
| ------------------- | ------------------------ | --------------------------------------- |
| CTC baseline        | ResNet-CTC               | Model sederhana, kuat, mudah dianalisis |
| CTC modern          | ConvNeXt-CTC             | Menguji peningkatan visual encoder      |
| Autoregressive      | TrOCR-Small aspect-aware | Model pretrained utama                  |
| Historical transfer | PyLaia atau Kraken       | Menguji keuntungan domain pretraining   |

Yang lain diposisikan sebagai berikut:

* CRNN awal: sanity-check baseline.
* TrOCR default: pembanding preprocessing.
* TrOCR-Base: dijalankan setelah Small stabil.
* TrOCR-Large: prioritas sangat rendah.
* PaddleOCR: cadangan atau diversity model.
* Ensemble: tahap akhir.

Dengan demikian, Anda tidak perlu secara aktif men-tuning semua sebelas workflow. Itu akan menghabiskan compute tanpa menghasilkan pemahaman yang cukup.

# Kesimpulan

Penjelasan sebelumnya **masih sepenuhnya relevan**. Bahkan setelah melihat seluruh workspace Anda, kesimpulannya menjadi lebih kuat:

> Anda tidak membutuhkan model tambahan. Anda membutuhkan proses eksperimen yang dapat menjelaskan mengapa sebuah model gagal dan perubahan apa yang benar-benar memperbaikinya.

Prioritasnya bukan sekadar “gonta-ganti hyperparameter”, melainkan:

1. memastikan evaluasi benar;
2. memastikan seluruh karakter dapat direpresentasikan;
3. membuktikan pipeline dapat menghafal subset kecil;
4. mengoptimalkan resolusi dan geometri line image;
5. memisahkan masalah training dari decoding;
6. melakukan error analysis per panjang line dan simbol;
7. baru melakukan tuning terkontrol;
8. menggunakan PyLaia/Kraken untuk domain transfer;
9. melakukan ensemble setelah model individual stabil.

Dengan workspace yang sekarang, kualitas hasil akan lebih banyak ditentukan oleh **disiplin eksperimen, data engineering, label engineering, decoding, dan error analysis** daripada jumlah arsitektur yang tersedia.

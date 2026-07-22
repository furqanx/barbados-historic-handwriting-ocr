Untuk konteks workspace dan kompetisi HTR Anda, saya membaginya menjadi **14 sisi engineering utama**. Ini bukan sekadar hyperparameter tuning, tetapi seluruh bagian pipeline yang dapat memengaruhi **loss, CER, WER, dan competition score**.

# 14 Sisi Engineering HTR

## 1. Data Integrity Engineering

Menjamin gambar dan transkripsi benar-benar valid.

Yang ditinjau:

* file rusak atau tidak dapat dibaca;
* gambar kosong atau hampir kosong;
* label kosong;
* ID duplikat;
* gambar duplikat dengan label berbeda;
* label salah ketik;
* line terpotong;
* ketidaksesuaian gambar dan transkripsi;
* sampel ekstrem atau anomalous.

Kesalahan di sini tidak dapat diperbaiki hanya dengan model lebih besar.

---

## 2. Dataset Split and Validation Engineering

Menentukan apakah validation score benar-benar mewakili test score.

Yang ditinjau:

* distribusi panjang teks antar-fold;
* distribusi simbol langka;
* distribusi writer atau handwriting style;
* distribusi kualitas gambar;
* distribusi aspect ratio;
* potensi duplicate leakage;
* stratification yang relevan;
* stabilitas skor lintas fold.

Bila satu writer muncul di train dan validation, validation score bisa terlalu optimistis.

---

## 3. Text Normalization Engineering

Menentukan bentuk teks yang dianggap sebagai target sebenarnya.

Yang ditinjau:

* Unicode NFC atau NFD;
* huruf kapital;
* spasi ganda;
* leading dan trailing whitespace;
* tab dan newline;
* apostrophe;
* quotation mark;
* hyphen, en dash, em dash;
* karakter yang secara visual sama tetapi berbeda Unicode;
* simbol seperti `^`, `*`, `~`, dan diakritik.

Normalisasi harus mengikuti aturan kompetisi. Jangan membersihkan simbol yang sebenarnya dinilai.

---

## 4. Charset and Tokenization Engineering

Memastikan semua karakter dapat direpresentasikan oleh model.

Untuk CTC:

* seluruh charset tersedia;
* blank index benar;
* unknown character tidak muncul;
* mapping karakter konsisten;
* encode-decode bersifat lossless.

Untuk TrOCR:

* simbol tersedia dalam tokenizer;
* tokenisasi karakter langka tidak menghasilkan urutan aneh;
* target tidak terpotong;
* special token tidak masuk prediksi;
* decoder dapat menghasilkan `^`, `*`, dan simbol lain.

Ini salah satu aspek terpenting untuk kompetisi character-sensitive.

---

## 5. Image Preprocessing Engineering

Mengoptimalkan visibilitas tulisan tanpa merusak informasi.

Yang dapat ditinjau:

* RGB versus grayscale;
* contrast enhancement;
* normalization;
* background normalization;
* denoising;
* sharpening ringan;
* binarization;
* deskew;
* cropping margin;
* removal of excessive whitespace;
* inversion bila tinta dan background terbalik.

Preprocessing agresif dapat menghilangkan titik, aksen, `*`, atau guratan tipis.

---

## 6. Image Geometry and Resolution Engineering

Mengatur bagaimana line panjang dimasukkan ke model.

Yang ditinjau:

* `target-height`;
* `max-width`;
* `canvas-width`;
* preservation of aspect ratio;
* padding;
* clipping;
* horizontal compression;
* pixels per character;
* image width setelah resize;
* width bucketing;
* batch padding efficiency.

Untuk dataset Anda, ini kemungkinan termasuk **tiga aspek paling berpengaruh** karena median aspect ratio sekitar 16,8.

---

## 7. Data Augmentation Engineering

Meningkatkan generalisasi tanpa menghancurkan karakter.

Augmentasi yang dapat diuji:

* brightness dan contrast;
* Gaussian noise;
* blur ringan;
* affine transformation;
* small rotation;
* elastic distortion;
* dilation dan erosion ringan;
* background variation;
* ink intensity variation;
* perspective distortion;
* random padding.

Yang harus dihindari:

* crop yang memotong karakter;
* blur berlebihan;
* erosion yang menghapus tanda baca;
* horizontal compression ekstrem;
* transformasi yang membuat label tidak lagi valid.

---

## 8. Sampling and Data-Balance Engineering

Mengatur sampel apa yang lebih sering dilihat model.

Yang dapat dilakukan:

* oversampling line dengan simbol langka;
* oversampling line panjang;
* writer-aware sampling;
* quality-aware sampling;
* curriculum dari line pendek ke panjang;
* balanced batch berdasarkan target length;
* rarity-aware sampler;
* hard-example sampling.

Contohnya, line yang mengandung `^` dan `*` dapat diberi probabilitas sampling lebih tinggi tanpa mengubah loss function.

---

## 9. Architecture and Feature-Extraction Engineering

Mengoptimalkan cara model mengekstrak representasi visual.

Pada workspace Anda, ini sudah tercakup melalui:

* CRNN;
* ResNet-style CNN;
* ConvNeXt;
* TrOCR encoder;
* PyLaia;
* Kraken;
* PaddleOCR/SVTR.

Hal yang tetap dapat ditinjau tanpa menambah arsitektur baru:

* horizontal stride;
* vertical stride;
* receptive field;
* feature-map height;
* sequence length;
* BiLSTM hidden size;
* number of recurrent layers;
* dropout;
* pretrained versus random initialization;
* layer freezing.

Fokusnya bukan menambah model, tetapi memastikan arsitektur yang ada tidak terlalu banyak mereduksi dimensi horizontal.

---

## 10. Sequence Alignment and Objective Engineering

Berkaitan dengan hubungan antara visual sequence dan target sequence.

### Untuk CTC

Yang diperiksa:

* encoder time steps;
* target length;
* adjacent repeated characters;
* valid CTC alignment;
* blank dominance;
* `input_lengths`;
* `target_lengths`;
* padding handling;
* `zero_infinity`;
* CTC collapse behavior.

### Untuk TrOCR

Yang diperiksa:

* teacher forcing;
* decoder start token;
* EOS;
* label masking;
* maximum target length;
* generation length;
* attention mask;
* exposure bias.

Loss yang turun tidak selalu berarti CER dan WER ikut membaik karena objective training berbeda dari proses decoding final.

---

## 11. Optimization and Training Engineering

Ini bagian yang biasanya disebut hyperparameter tuning.

Yang ditinjau:

* learning rate;
* optimizer;
* weight decay;
* batch size;
* gradient accumulation;
* gradient clipping;
* warmup;
* scheduler;
* number of epochs;
* early stopping;
* mixed precision;
* checkpoint selection;
* frozen versus unfrozen layers;
* differential learning rate;
* random seed.

Namun aspek ini seharusnya dilakukan **setelah** data, charset, geometri, dan alignment sudah benar.

---

## 12. Transfer Learning and Domain Adaptation Engineering

Memanfaatkan pretrained model secara tepat.

Yang ditinjau:

* ImageNet versus IAM pretraining;
* historical HTR checkpoint;
* kemiripan bahasa;
* kemiripan handwriting style;
* kemiripan periode dokumen;
* kemiripan charset;
* kemiripan transcription convention;
* full fine-tuning;
* partial freezing;
* gradual unfreezing;
* learning rate berbeda untuk backbone dan decoder.

PyLaia dan Kraken berada terutama dalam aspek ini. Nilai mereka bukan hanya arsitektur, tetapi domain knowledge yang sudah dipelajari.

---

## 13. Decoding and Post-processing Engineering

Mengubah probabilitas model menjadi teks final.

### CTC

* greedy decoding;
* beam search;
* beam width;
* character language model;
* lexicon-free decoding;
* blank handling;
* repeated-character handling.

### TrOCR

* greedy generation;
* beam search;
* length penalty;
* maximum generation length;
* EOS behavior;
* repetition penalty;
* sequence score.

### Post-processing

* menghapus special token;
* mempertahankan simbol asli;
* mengontrol whitespace;
* memperbaiki format tanpa mengubah isi;
* tidak melakukan spell correction yang merusak transkripsi literal.

Decoding yang buruk dapat membuat model dengan loss bagus menghasilkan CER buruk.

---

## 14. Evaluation, Error Analysis, and Ensemble Engineering

Ini merupakan tahap untuk mengetahui eksperimen mana yang benar-benar bekerja.

Yang ditinjau:

* CER;
* WER;
* competition score;
* insertion;
* deletion;
* substitution;
* character recall;
* symbol recall;
* CER berdasarkan panjang line;
* CER berdasarkan aspect ratio;
* CER pada line dengan simbol langka;
* CER berdasarkan kualitas gambar;
* per-writer performance;
* confusion matrix karakter;
* out-of-fold score;
* variance antar-fold.

Setelah itu barulah dilakukan ensemble berdasarkan error diversity:

* CTC + TrOCR;
* ConvNeXt-CTC + Kraken;
* TrOCR + PyLaia;
* beberapa fold model yang sama;
* model dengan kekuatan berbeda pada line panjang dan simbol langka.

---

# Pengelompokan Besar

Keempat belas sisi tersebut dapat diringkas menjadi lima kelompok.

| Kelompok                         | Sisi engineering |
| -------------------------------- | ---------------- |
| **Data**                         | 1–4              |
| **Visual input**                 | 5–8              |
| **Model dan training**           | 9–12             |
| **Output**                       | 13               |
| **Measurement dan final system** | 14               |

# Tingkat Prioritas untuk Workspace Anda

## Prioritas P0: harus diaudit terlebih dahulu

1. Data integrity
2. Text normalization
3. Charset dan tokenization
4. Image geometry dan resolution
5. Sequence alignment
6. Evaluation consistency

Jika bagian ini salah, eksperimen hyperparameter tidak dapat dipercaya.

## Prioritas P1: kemungkinan memberi peningkatan besar

7. Sampling dan rare-symbol handling
8. Augmentation
9. Optimization
10. Transfer learning PyLaia/Kraken
11. Decoding

## Prioritas P2: dilakukan setelah model individual stabil

12. Arsitektur lanjutan dalam workflow yang sama
13. Multi-fold confirmation
14. Ensemble dan post-processing

# Inti Jawaban

Jadi, untuk kompetisi Anda terdapat **14 sisi engineering yang perlu ditinjau**:

1. Data integrity
2. Dataset split dan validation
3. Text normalization
4. Charset dan tokenization
5. Image preprocessing
6. Image geometry dan resolution
7. Data augmentation
8. Sampling dan balancing
9. Architecture dan feature extraction
10. Sequence alignment dan objective
11. Optimization dan training
12. Transfer learning dan domain adaptation
13. Decoding dan post-processing
14. Evaluation, error analysis, dan ensemble

Dari semuanya, peningkatan terbesar kemungkinan bukan berasal dari menambah model, melainkan dari kombinasi:

> **geometri input + charset/symbol handling + alignment + sampling + decoding + error analysis.**

Hyperparameter tuning hanyalah **satu dari empat belas sisi tersebut**, bukan keseluruhan engineering proses pelatihan.

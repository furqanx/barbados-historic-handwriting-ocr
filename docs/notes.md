Durasi kompetisi: 3 Juli 2026 sampai 4 Oktober 2026.
Itu sekitar 94 hari kalender jika dihitung inklusif.
Dengan batas 200 total submission, rata-rata aman:
2 submission per hari, sesekali boleh 3 submission untuk eksperimen penting.









---

Implikasinya untuk modeling:

- Ini lebih cocok diperlakukan sebagai **handwritten line recognition**, bukan word-level OCR.
- Model perlu bisa menghasilkan sequence teks cukup panjang.
- Baseline yang cocok: model OCR/HTR encoder-decoder atau CRNN/TrOCR-style, bukan image classifier.
- Untuk preprocessing, kita perlu hati-hati menjaga aspect ratio gambar karena panjang baris teks bervariasi.
- Metric CER akan sangat penting, karena kesalahan kecil per karakter bisa banyak terakumulasi pada teks sepanjang ini.

--- 

**Handwritten line recognition** adalah task OCR yang input-nya **satu gambar berisi satu baris/frasa tulisan tangan**, lalu output-nya **urutan teks lengkap**.

Beda dengan image classification:

```text
image -> class label
```

Line recognition:

```text
image of handwritten line -> "By this public act and Instrument of protest"
```

Model biasanya bekerja sebagai **sequence prediction**.



**Contoh Arsitektur Umum**
1. **CRNN + CTC**
   - CNN membaca fitur visual dari gambar.
   - RNN/Transformer membaca urutan fitur dari kiri ke kanan.
   - CTC loss mencocokkan urutan karakter tanpa perlu tahu posisi setiap huruf.
   - Cocok untuk OCR klasik.

   Contoh output:
   ```text
   image -> characters -> "Signed Sealed and delivered in the presence of us"
   ```

2. **TrOCR / Vision Encoder-Decoder**
   - Encoder visual membaca gambar.
   - Decoder bahasa menghasilkan teks token demi token.
   - Lebih modern, sering memakai pretrained Transformer.
   - Cocok untuk handwriting recognition jika pretrained model tersedia.

3. **Donut / OCR-free Document Model**
   - Image masuk ke Transformer.
   - Model langsung menghasilkan teks.
   - Biasanya kuat untuk dokumen, tapi bisa lebih berat.
  
4. **Model pretrained OCR/HTR dari Hugging Face jika cocok.**




Untuk kompetisi ini, karena target rata-rata **11 kata / 62 karakter**, pendekatan yang masuk akal:
- baseline cepat: pakai pretrained **TrOCR handwritten**
- lalu fine-tune pada `Train.csv`
- evaluasi lokal dengan CER/WER
- generate `Target` untuk test set

Contoh konsep minimal:

```text
image line crop
   ↓
TrOCR processor
   ↓
VisionEncoderDecoderModel
   ↓
predicted text
```

Jadi kita tidak memprediksi satu label, tapi memprediksi **kalimat/frasa hasil transkripsi**.




Bisa dipakai **secara berurutan sebagai strategi eksperimen**, tapi keempatnya bukan “tahapan” dalam satu pipeline. Mereka adalah **keluarga pendekatan/model yang berbeda** untuk task OCR/HTR.

Ringkasnya:

1. **CRNN + CTC**
   Baseline klasik dan paling cocok untuk mulai dari nol. Input gambar baris teks, output karakter. Ringan, mudah dikontrol, cocok untuk dataset kecil-menengah.

2. **TrOCR / Vision Encoder-Decoder**
   Pendekatan modern berbasis Transformer. Biasanya memakai pretrained model, lalu fine-tuning. Potensinya lebih kuat, tapi lebih berat dan butuh handling tokenizer/processor.

3. **Donut**
   Lebih cocok untuk dokumen utuh/structured extraction. Untuk line-level handwriting, bisa digunakan, tapi mungkin terlalu berat dan tidak selalu paling natural.

4. **Pretrained OCR/HTR Hugging Face**
   Ini bukan metode arsitektur tunggal, tapi kategori: bisa berisi TrOCR, CRNN, PARSeq, ViTSTR, atau model HTR lain yang sudah pretrained.

Jadi urutan eksperimen yang masuk akal:

1. **CRNN + CTC baseline**
   Untuk memastikan pipeline data, metric, training, inference, dan submission berjalan.

2. **Fine-tune pretrained HTR/OCR model**
   Cari model yang memang dekat dengan handwriting/line recognition.

3. **TrOCR fine-tuning**
   Jika pretrained TrOCR cocok dan resource cukup.

4. **Donut**
   Prioritas lebih rendah untuk kompetisi ini, kecuali nanti EDA visual menunjukkan format dokumennya lebih kompleks dari sekadar line image.

Menurut saya: **mulai dari CRNN + CTC dulu**, lalu naik ke pretrained model. CRNN bukan target akhir yang pasti menang, tapi sangat bagus sebagai “mesin baseline” untuk mengunci workflow kompetisi.





---
Kesimpulan utama:
- Semua gambar **terbaca dengan baik**: 5.472 readable, **0 corrupt**.
- Semua gambar berformat **RGB**, jadi preprocessing bisa konsisten.
- Gambar berbentuk **sangat horizontal/panjang**:
  - median aspect ratio sekitar **16.8**
  - ini cocok dengan data **line-level handwriting**, bukan crop kata tunggal.
- Ukuran gambar sangat bervariasi:
  - width: **267 sampai 6051 px**
  - height: **28 sampai 1131 px**
- Ada dua kelompok besar kemungkinan:
  - crop pendek/kecil: median sekitar **1119 x 65**
  - crop besar/tinggi: banyak di atas **2800 x 214**
- Karena variasi ukuran besar, preprocessing penting:
  - resize berdasarkan tinggi tetap
  - pertahankan aspect ratio
  - padding/truncation untuk width
- Jangan resize langsung ke ukuran kotak kecil, karena teks panjang bisa terdistorsi dan merusak OCR.

---
Kesimpulannya:
- Distribusi ukuran gambar **tidak tunggal**, ada dua kelompok jelas:
  - mayoritas gambar kecil/pendek sekitar **width 1000-1200 px**, **height 40-100 px**
  - sebagian gambar besar sekitar **width 3000-5000 px**, **height 250-400 px**
- Ini menandakan data kemungkinan berasal dari **crop dengan resolusi/skala berbeda**, bukan hasil preprocessing yang seragam.
- Aspect ratio mayoritas tetap tinggi, sekitar **10-25**, jadi hampir semua memang gambar baris teks horizontal.
- Untuk training, preprocessing harus menormalkan skala:
  - resize height ke nilai tetap, misalnya 64 atau 96
  - width mengikuti aspect ratio
  - lalu pad/crop ke max width
- Kita perlu cek apakah kelompok kecil vs besar berbeda antara train/test, karena kalau distribusinya beda bisa memengaruhi performa.

---
Kesimpulannya:
- Semua train/test punya gambar: **0 missing image**, data aman untuk training dan inference.
- Ada **dua cluster ukuran gambar**:
  - cluster kecil sekitar width **1000-1300**
  - cluster besar sekitar width **3500-6000**
- Panjang target cukup berkorelasi dengan jumlah kata: `char_len` vs `word_len` = **0.84**, normal.
- Korelasi `width` dengan `char_len` lemah negatif (**-0.12**), jadi **lebar gambar tidak langsung mencerminkan panjang teks** karena skala/resolusi berbeda.
- `aspect_ratio` lebih berkorelasi dengan panjang teks (**0.38**) dibanding width mentah, jadi aspect ratio lebih informatif.
- Ada kemungkinan gambar berasal dari **dua pipeline crop/resolution berbeda**.

Implikasi praktis: preprocessing harus **normalisasi tinggi/scale** dulu, bukan memakai ukuran asli mentah. Train/valid split juga sebaiknya mempertahankan distribusi ukuran/aspect ratio.

---
Implikasi:
- Tidak ada indikasi kuat bahwa test lebih blur, lebih gelap, atau lebih noisy daripada train.
- Preprocessing yang sama untuk train/test seharusnya aman.
- Karena ada dua cluster blur/sharpness dan dua cluster ukuran, augmentasi ringan seperti blur, contrast jitter, brightness jitter bisa berguna saat training.
- Kita tetap perlu hati-hati agar preprocessing tidak menghapus tinta tipis, karena banyak gambar punya background terang dan dark pixel ratio rendah.
- Simpulan singkat: train-test matching bagus secara kualitas visual; tantangan utama bukan domain shift, tapi variasi skala, handwriting style, dan transkripsi historis.
- Kita tetap perlu preprocessing yang robust terhadap dua skala gambar, tapi tidak perlu strategi khusus yang memisahkan train dan test.






























---
## EDA Summary - Initial Notebook

Notebook: `notebooks/exploration/01_initial_eda.ipynb`

### Data structure

- `Train.csv`: 4.098 rows, 2 columns (`ID`, `Target`)
- `Test.csv`: 1.374 rows, 1 column (`ID`)
- `SampleSubmission.csv`: 1.374 rows, 2 columns (`ID`, `Target`)
- Tidak ada missing value pada `Train.csv` dan `Test.csv`.
- Tidak ada duplicate ID pada train, test, maupun sample submission.
- Sample submission IDs sama persis dengan test IDs.

### Image availability

- Total images: 5.472
- Train images missing: 0
- Test images missing: 0
- Extra images not referenced by train/test: 0
- Corrupt/bad images: 0
- Semua image mode: RGB

### Target text

- Semua train rows punya target teks.
- Target unik: 4.086 dari 4.098 rows.
- Duplicate exact target rows: 12.
- Ini berarti dataset hampir tidak repetitif; model perlu benar-benar membaca gambar, bukan menghafal frasa.

Target length:

- Median: 62 karakter, 11 kata.
- Mean: 62,28 karakter, 11,28 kata.
- Min: 37 karakter, 4 kata.
- Max: 120 karakter, 22 kata.
- 95% target <= 85 karakter dan <= 16 kata.

Implikasi:

- Task ini adalah **handwritten line recognition**, bukan word-level OCR.
- Output berupa sequence teks/frasa cukup panjang.
- Baseline cocok: TrOCR-style encoder-decoder atau CRNN/CTC.

### Character and vocabulary

- Unique characters: 82.
- Karakter mencakup huruf besar/kecil, angka, spasi, punctuation, dan simbol historis/markup seperti `^`, `~`, `&`, `:`, `-`.
- Kata paling sering mencerminkan dokumen legal/historis: `and`, `the`, `of`, `to`, `or`, `said`, `heires`, `assignes`, `barbados`, dll.

Implikasi:

- Jangan lower-case atau menghapus punctuation secara agresif sebelum tahu aturan scoring.
- Case, simbol, dan ejaan historis kemungkinan penting untuk CER/WER.

### Image dimensions

- Width sangat bervariasi: 267 sampai 6.051 px.
- Height sangat bervariasi: 28 sampai 1.131 px.
- Median size: sekitar 1.119 x 65 px.
- Median aspect ratio: 16,78.
- Mayoritas gambar sangat horizontal, sesuai line-level handwriting.
- Ada dua cluster ukuran:
  - cluster kecil: sekitar width 1.000-1.300 dan height 40-100.
  - cluster besar: sekitar width 3.500-6.000 dan height 250-400.

Implikasi:

- Preprocessing perlu resize berdasarkan height tetap, menjaga aspect ratio.
- Hindari resize langsung ke square image.
- Perlu padding/truncation width.
- Train/validation split sebaiknya mempertahankan distribusi aspect ratio/ukuran.

### Correlations

- `char_len` vs `word_len`: 0,841, sesuai ekspektasi.
- `width` vs `char_len`: -0,124, lemah dan negatif.
- `height` vs `char_len`: -0,226, lemah dan negatif.
- `aspect_ratio` vs `char_len`: 0,380.

Implikasi:

- Width mentah tidak bisa dipakai sebagai proxy panjang teks karena skala gambar tidak konsisten.
- Aspect ratio lebih informatif daripada width mentah.
- Normalisasi skala image penting sebelum training.

### Baseline direction

Baseline pertama yang masuk akal:

1. Buat split train/valid yang stratified-ish berdasarkan target length dan image size/aspect ratio.
2. Implement metric lokal CER/WER.
3. Coba pretrained OCR/HTR model seperti TrOCR handwritten.
4. Preprocessing image:
   - convert RGB konsisten,
   - resize height tetap,
   - keep aspect ratio,
   - pad width.
5. Generate baseline predictions dan submission.

---

## EDA Summary - Train/Test Image and Text Audit

Notebook: `notebooks/exploration/02_train_test_image_and_text_audit.ipynb`

### Train vs test image distribution

Train/test sangat mirip secara distribusi ukuran gambar.

- Train rows: 4.098
- Test rows: 1.374
- Median width:
  - train: 1.118 px
  - test: 1.120 px
- Median height:
  - train: 65 px
  - test: 65 px
- Median aspect ratio:
  - train: 16,77
  - test: 16,82
- Median area:
  - train: 69.392,5 px
  - test: 69.564 px

Visual histogram menunjukkan:

- Kedua split punya dua cluster ukuran yang sama:
  - cluster kecil sekitar width 1.000-1.300 px.
  - cluster besar sekitar width 3.500-5.500 px.
- Test sedikit lebih dominan pada cluster kecil.
- Train punya beberapa extreme image yang lebih besar, tetapi tidak terlihat domain shift besar.

Implikasi:

- Validasi lokal dari train kemungkinan cukup representatif untuk test.
- Fold berdasarkan target length + aspect ratio + area tetap masuk akal.
- Tidak perlu split strategy khusus untuk domain train/test yang berbeda.

### Train vs test image quality

Quality signals train/test juga sangat mirip.

- Median brightness mean:
  - train: 191,32
  - test: 191,53
- Median contrast std:
  - train: 31,83
  - test: 32,07
- Median blur/sharpness Laplacian variance:
  - train: 1.433,69
  - test: 1.442,70

Visual histogram menunjukkan:

- Brightness train/test overlap kuat.
- Contrast train/test overlap kuat.
- Blur/sharpness punya dua cluster di kedua split:
  - cluster blurrier sekitar 200-500.
  - cluster sharper sekitar 1.300-2.300.

Implikasi:

- Tidak ada indikasi test lebih gelap, lebih blur, atau lebih noisy daripada train.
- Preprocessing yang sama untuk train/test aman.
- Augmentasi ringan seperti brightness/contrast jitter dan mild blur bisa berguna.
- Jangan terlalu agresif thresholding/denoising karena tinta tipis bisa hilang.

### Target normalization audit

Ada beberapa isu whitespace dan banyak sinyal bahwa case/punctuation penting.

- Leading space rows: 5
- Trailing space rows: 95
- Repeated whitespace rows: 55
- Rows containing uppercase: 3.164
- Rows containing digits: 217
- Rows containing punctuation/symbols: 1.841

Unique target count under normalization variants:

- Original unique: 4.086
- Strip unique: 4.085
- Collapse whitespace unique: 4.085
- Lowercase unique: 4.076
- Lowercase + collapse whitespace unique: 4.074

Observed special characters include punctuation and historical transcription markers:

- `^`, `~`, `&`, `:`, `;`, `,`, `.`, `-`, `(`, `)`, `+`, `#`, `|`, `\\`, `_`, `?`, `"`, `'`, `*`

Implikasi:

- Untuk metric/evaluasi lokal, whitespace boleh dinormalisasi secara ringan agar prediksi tidak dihukum karena accidental extra spaces.
- Untuk target training dan submission, jangan lower-case.
- Jangan hapus punctuation/symbol secara default.
- Case, digit, punctuation, dan simbol historis kemungkinan harus dipelajari model.
- Normalisasi yang aman untuk baseline: `strip` + collapse repeated whitespace, tetapi preserve case dan punctuation.

### Code implications

Fondasi kode saat ini sudah cukup untuk baseline pertama:

- constants sudah ada.
- metric CER/WER sudah ada dan default-nya melakukan whitespace normalization ringan.
- manifest/fold sudah ada dan distribusi fold seimbang.
- dataset wrapper sudah bisa membaca image + target.

Tambahan yang akan berguna sebelum training model:

1. Modul preprocessing image yang configurable:
   - fixed target height,
   - keep aspect ratio,
   - max width,
   - padding color,
   - optional train augmentations.
2. Modul text normalization yang reusable:
   - preserve case/punctuation by default,
   - strip + collapse whitespace,
   - mode berbeda untuk metric/training/submission bila diperlukan.

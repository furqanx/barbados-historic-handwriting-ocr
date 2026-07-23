# EDA Summary

Ringkasan ini berasal dari:

- `notebooks/exploration/01_initial_eda.ipynb`
- `notebooks/exploration/02_train_test_image_and_text_audit.ipynb`
- `notebooks/exploration/03_data_integrity_engineering.ipynb`

Tujuannya adalah menjaga keputusan modeling tetap sesuai dengan bentuk data kompetisi.

## Competition Duration

Kompetisi berjalan dari 3 Juli 2026 sampai 4 Oktober 2026, sekitar 94 hari kalender jika dihitung inklusif.

Batas submission:

- Maksimal 5 submission per hari.
- Maksimal 200 submission total.

Ritme aman: sekitar 2 submission per hari, dengan ruang untuk 3 submission pada eksperimen penting.

## Data Structure

- `Train.csv`: 4.098 rows, kolom `ID` dan `Target`.
- `Test.csv`: 1.374 rows, kolom `ID`.
- `SampleSubmission.csv`: 1.374 rows, kolom `ID` dan `Target`.
- Tidak ada missing value pada `Train.csv` dan `Test.csv`.
- Tidak ada duplicate ID pada train, test, maupun sample submission.
- Sample submission IDs sama dengan test IDs.

## Image Availability

- Total images: 5.472.
- Train images missing: 0.
- Test images missing: 0.
- Extra images not referenced by train/test: 0.
- Corrupt/bad images: 0.
- Semua image mode: RGB.

Implikasi: pipeline data aman untuk training dan inference; fokus utama bukan memperbaiki file rusak, tetapi preprocessing dan recognition quality.

## Task Type

Kompetisi ini adalah **handwritten line recognition**.

Input berupa satu gambar berisi satu baris/frasa tulisan tangan, dan output berupa transkripsi lengkap:

```text
image line crop -> "By this public act and Instrument of protest"
```

Ini bukan image classification dan bukan word-level OCR. Model harus menghasilkan sequence teks yang cukup panjang.

Pendekatan yang sesuai:

- CRNN/ResNet/ConvNeXt + CTC.
- TrOCR atau vision encoder-decoder.
- Historical HTR model seperti PyLaia atau Kraken jika domain cocok.

Pendekatan yang kurang natural:

- Donut/document understanding untuk dokumen utuh.
- Scene text recognizer yang didesain untuk crop kata pendek.

## Target Text

- Empty target rows: 0.
- Unique exact targets: 4.086 dari 4.098 rows.
- Duplicated exact target rows: 12.

Panjang target:

- Mean: 62,28 karakter dan 11,28 kata.
- Median: 62 karakter dan 11 kata.
- Min: 37 karakter dan 4 kata.
- Max: 120 karakter dan 22 kata.
- 95% target <= 85 karakter dan <= 16 kata.

Implikasi:

- Dataset hampir tidak repetitif.
- Model harus membaca gambar, bukan menghafal frasa.
- Sequence length handling sangat penting, terutama untuk CTC.

## Character And Vocabulary

- Unique characters: 82.
- Karakter mencakup huruf besar/kecil, angka, spasi, punctuation, dan simbol historis/markup.
- Simbol penting yang muncul: `^`, `~`, `&`, `:`, `;`, `,`, `.`, `-`, `(`, `)`, `+`, `#`, `|`, `\\`, `_`, `?`, `"`, `'`, `*`.
- Kata yang sering muncul mencerminkan dokumen legal/historis: `and`, `the`, `of`, `to`, `or`, `said`, `heires`, `assignes`, `barbados`.

Implikasi:

- Jangan lower-case target secara default.
- Jangan menghapus punctuation/symbol secara agresif.
- Normalisasi aman untuk baseline: `strip` + collapse repeated whitespace, sambil mempertahankan case dan punctuation.

## Image Dimensions

Ukuran image sangat bervariasi:

- Width: 267 sampai 6.051 px.
- Height: 28 sampai 1.131 px.
- Median size: sekitar 1.119 x 65 px.
- Median aspect ratio: 16,78.

Ada dua cluster ukuran:

- Cluster kecil: width sekitar 1.000-1.300 px, height 40-100 px.
- Cluster besar: width sekitar 3.500-6.000 px, height 250-400 px.

Implikasi:

- Resize harus berbasis fixed height.
- Aspect ratio harus dipertahankan.
- Hindari resize langsung ke square image.
- Gunakan padding/truncation width atau width bucketing.

## Correlation Notes

- `char_len` vs `word_len`: 0,841.
- `width` vs `char_len`: -0,124.
- `height` vs `char_len`: -0,226.
- `aspect_ratio` vs `char_len`: 0,380.

Width mentah tidak bisa menjadi proxy panjang teks karena skala gambar tidak konsisten. Aspect ratio lebih informatif, tetapi tetap bukan pengganti label length.

## Train Vs Test Distribution

Train/test mirip secara ukuran gambar:

- Median width train: 1.118 px.
- Median width test: 1.120 px.
- Median height train: 65 px.
- Median height test: 65 px.
- Median aspect ratio train: 16,77.
- Median aspect ratio test: 16,82.

Kedua split memiliki dua cluster ukuran yang sama. Test sedikit lebih dominan di cluster kecil, tetapi tidak terlihat domain shift besar.

Implikasi:

- Validasi lokal dari train kemungkinan cukup representatif untuk test.
- Fold berdasarkan target length + aspect ratio + area tetap masuk akal.

## Train Vs Test Image Quality

Quality signals train/test mirip:

- Median brightness mean train: 191,32.
- Median brightness mean test: 191,53.
- Median contrast std train: 31,83.
- Median contrast std test: 32,07.
- Median blur/sharpness train: 1.433,69.
- Median blur/sharpness test: 1.442,70.

Implikasi:

- Tidak ada indikasi test lebih gelap, lebih blur, atau lebih noisy daripada train.
- Preprocessing yang sama untuk train/test aman.
- Augmentasi ringan seperti brightness/contrast jitter dan mild blur bisa berguna.
- Hindari thresholding/denoising terlalu agresif karena tinta tipis bisa hilang.

## Target Normalization Audit

Observed issues:

- Leading space rows: 5.
- Trailing space rows: 95.
- Repeated whitespace rows: 55.
- Rows containing uppercase: 3.164.
- Rows containing digits: 217.
- Rows containing punctuation/symbols: 1.841.

Unique target count:

- Original unique: 4.086.
- Strip unique: 4.085.
- Collapse whitespace unique: 4.085.
- Lowercase unique: 4.076.
- Lowercase + collapse whitespace unique: 4.074.

Implikasi:

- Whitespace cleanup ringan aman.
- Lowercase tidak aman sebagai default.
- Case, digit, punctuation, dan simbol historis perlu dipelajari model.

## Data Integrity Audit

Hal yang sudah diaudit:

- File rusak/tidak terbaca.
- Gambar kosong/hampir kosong.
- Label kosong.
- ID duplikat.
- Gambar duplikat dengan label berbeda.
- Sampel ekstrem/anomalous.
- Ketidaksesuaian image dan label secara heuristik.

Tidak ditemukan blocker besar pada integritas file. Isu yang tersisa lebih banyak berupa sample sulit, label dengan simbol langka, whitespace, dan variasi visual.

## Modeling Implications

Baseline yang paling masuk akal:

1. Split train/valid yang menjaga distribusi length dan image geometry.
2. Metric lokal CER/WER yang konsisten dengan submission.
3. Image preprocessing:
   - RGB konsisten,
   - fixed target height,
   - preserve aspect ratio,
   - pad/crop max width.
4. Text normalization:
   - preserve case/punctuation,
   - strip + collapse whitespace untuk metric/prediction cleanup.
5. Validasi pipeline:
   - charset/tokenizer round-trip,
   - CTC alignment audit,
   - evaluator consistency,
   - small-subset overfit test.

Kesimpulan utama: data cukup bersih, train/test cukup matching, dan tantangan utama adalah sequence recognition pada historical handwriting dengan variasi skala dan simbol target yang tidak boleh disederhanakan sembarangan.

# Konklusi Strategi Kompetisi Historical Handwritten Line Recognition

## 1. Karakteristik sebenarnya dari task

Task ini bukan OCR modern biasa. Model harus melakukan **transkripsi literal** terhadap satu baris dokumen historis dengan karakteristik:

* ejaan historis dan kata yang tidak mengikuti bahasa modern;
* bentuk huruf yang sangat ambigu;
* singkatan dan simbol seperti `^`, `*`, apostrof, dan tanda baca;
* kapitalisasi yang tidak konsisten;
* potongan kalimat yang tidak lengkap;
* bagian kertas rusak atau hilang;
* multiple spaces yang mungkin merepresentasikan jarak fisik pada dokumen;
* gambar sangat lebar dengan median aspect ratio sekitar 16,8.

Konsekuensinya, sistem tidak boleh sekadar menghasilkan kalimat yang “masuk akal”. Sistem harus menghasilkan **teks yang sedekat mungkin dengan anotasi literal**, termasuk karakter dan whitespace.

---

# 2. Kebijakan label dan normalisasi teks

Ini harus ditetapkan sebelum eksperimen model lebih lanjut.

## Yang harus dipertahankan

* huruf besar dan kecil;
* ejaan historis;
* simbol khusus;
* tanda baca;
* urutan karakter;
* multiple spaces apabila evaluator menghitungnya;
* bentuk asli kata yang salah menurut ejaan modern.

Contoh:

```text
Sell alien confirme assigne and   unto him the said Nicolas Barnes
```

Tidak boleh diubah menjadi:

```text
Sell, alien, confirm, assign and unto him the said Nicolas Barnes
```

## Yang perlu diaudit

* Unicode NFC versus NFD;
* karakter yang tampak sama tetapi berbeda code point;
* apostrof lurus dan apostrof melengkung;
* hyphen, en dash, dan em dash;
* regular space versus non-breaking space;
* tab, newline, atau carriage return tersembunyi;
* leading dan trailing whitespace;
* jumlah spasi berurutan.

Buat satu fungsi normalisasi resmi yang digunakan bersama oleh:

* preprocessing label;
* tokenizer;
* training;
* decoding;
* evaluator;
* converter submission.

Normalisasi hanya boleh memperbaiki representasi teknis. Jangan membersihkan isi label.

---

# 3. Audit evaluator sebelum optimasi

Anda harus mengetahui dengan pasti bagaimana competition score menghitung whitespace.

Periksa apakah:

* CER menghitung setiap spasi sebagai karakter;
* WER menggunakan `split()` sehingga beberapa spasi dianggap satu separator;
* punctuation dihitung sebagai bagian kata;
* case sensitivity diterapkan;
* Unicode dinormalisasi;
* leading dan trailing spaces diabaikan atau dihitung.

Gunakan evaluator lokal yang identik dengan evaluator kompetisi. Semua checkpoint harus dipilih berdasarkan:

1. competition score;
2. CER;
3. WER;
4. validation loss hanya sebagai indikator tambahan.

**Loss bukan metrik utama untuk memilih model final.**

---

# 4. Preprocessing gambar

## Prinsip utama

Preprocessing harus meningkatkan keterbacaan tinta tanpa menghilangkan informasi kecil atau mengubah geometri horizontal.

## Rekomendasi utama

### Preserve aspect ratio

Gunakan fixed height, tetapi pertahankan rasio lebar gambar.

Eksperimen utama:

* tinggi 64;
* tinggi 96;
* tinggi 128.

Tinggi 96 atau 128 kemungkinan lebih aman untuk:

* titik;
* aksen;
* `^`;
* `*`;
* garis pendek;
* perbedaan antara huruf yang bentuknya mirip.

### Width handling

Gunakan:

* dynamic width;
* width bucketing;
* padding ke kanan;
* attention mask atau input length yang benar;
* `max-width` yang tidak terlalu agresif.

Catat berapa persen gambar yang:

* dipotong;
* diperkecil secara ekstrem;
* melewati batas canvas;
* memiliki pixel-per-character terlalu rendah.

Gunakan indikator:

[
\text{pixels per character}
===========================

\frac{\text{resized image width}}{\text{target character length}}
]

Line panjang dengan terlalu sedikit pixel per karakter akan sulit mengenali tanda kecil.

## Internal gap tidak boleh dihapus

Pada contoh gambar, bagian kertas yang hilang berkaitan dengan multiple spaces pada label.

Karena itu:

* jangan menghapus whitespace internal;
* jangan menyambungkan dua bagian tulisan yang terpisah;
* jangan melakukan horizontal compression ekstrem;
* jangan crop bagian kosong di tengah line;
* jangan melakukan transformasi yang mengubah panjang gap secara besar.

External margin boleh dipotong secara konservatif, tetapi struktur horizontal internal harus dipertahankan.

## Warna dan kontras

Untuk model pretrained seperti TrOCR dan ConvNeXt:

* pertahankan RGB sebagai baseline utama;
* gunakan normalization sesuai checkpoint;
* bandingkan dengan grayscale yang dikonversi menjadi tiga channel bila perlu.

Augmentasi atau enhancement yang aman:

* contrast adjustment ringan;
* brightness adjustment ringan;
* background intensity variation;
* noise ringan;
* blur ringan;
* sharpening sangat ringan.

Hindari sebagai default:

* binarization keras;
* aggressive denoising;
* erosion kuat;
* crop acak;
* elastic distortion ekstrem;
* deskew yang berlebihan.

Metode agresif dapat menghapus tanda baca dan guratan tipis.

---

# 5. Feature engineering

Untuk arsitektur deep learning yang Anda gunakan, **handcrafted feature seperti HOG, SIFT, atau projection profile tidak perlu menjadi jalur utama**.

Feature engineering yang penting justru berada pada representasi input dan sequence.

## Untuk model CTC

Pastikan encoder:

* mempertahankan resolusi horizontal;
* tidak menggunakan horizontal stride terlalu besar;
* menghasilkan cukup time step;
* mereduksi tinggi feature map hingga menjadi sequence horizontal;
* menggunakan panjang sequence asli, bukan panjang padding.

Untuk target maksimum sekitar 120 karakter, horizontal downsampling harus cukup kecil agar alignment CTC tetap valid.

Periksa:

[
T \geq U + R
]

dengan:

* (T): jumlah encoder time step;
* (U): panjang target;
* (R): jumlah pasangan karakter identik yang berurutan.

## Untuk TrOCR

Fokus feature engineering ada pada:

* patch resolution;
* target height;
* canvas width;
* preserve-ratio preprocessing;
* positional information;
* attention mask;
* jumlah visual token.

TrOCR default square-resize harus tetap menjadi pembanding, tetapi varian aspect-ratio-aware seharusnya menjadi jalur utama.

---

# 6. Augmentation

Augmentation harus mensimulasikan variasi dokumen historis, bukan sekadar menambah distorsi.

## Augmentasi yang layak diuji

* brightness;
* contrast;
* gamma;
* noise ringan;
* blur ringan;
* small rotation;
* small affine transformation;
* ink fading;
* background variation;
* dilation atau erosion sangat ringan.

## Augmentasi yang harus dikontrol ketat

* perspective distortion;
* elastic transformation;
* horizontal scaling;
* cropping;
* strong blur;
* aggressive erosion.

Gunakan augmentasi lebih ringan pada line yang mengandung simbol langka. Simbol seperti `^` dan `*` mudah hilang akibat blur atau erosion.

Augmentasi harus diuji sebagai ablation:

```text
tanpa augmentasi
→ augmentasi ringan
→ augmentasi sedang
```

Jangan langsung menggunakan seluruh transformasi sekaligus.

---

# 7. Sampling dan penanganan karakter langka

Buat statistik:

* frekuensi setiap karakter;
* jumlah line yang mengandung setiap karakter;
* frekuensi multiple spaces;
* distribusi panjang target;
* distribusi line dengan kerusakan fisik;
* distribusi aspect ratio.

## Strategi sampling

Gunakan weighted sampling atau oversampling ringan untuk:

* line dengan `^`;
* line dengan `*`;
* line dengan punctuation langka;
* line sangat panjang;
* line dengan multiple spaces;
* line dengan kualitas visual buruk.

Jangan oversample terlalu ekstrem karena dapat menyebabkan insertion error, misalnya model mulai menambahkan `*` pada line yang sebenarnya tidak memilikinya.

Lebih aman meningkatkan peluang line langka sekitar beberapa kali, kemudian memantau:

* recall karakter;
* precision karakter;
* insertion count.

---

# 8. Strategi pemilihan model

Anda tidak perlu menambah workflow baru. Dari model yang sudah tersedia, fokus aktif sebaiknya dipersempit.

## Jalur utama 1: ResNet-CTC

Peran:

* baseline kuat dan stabil;
* mudah dianalisis;
* literal terhadap karakter;
* relatif ringan.

Model ini menjadi referensi utama untuk:

* geometri input;
* charset;
* alignment;
* decoding CTC;
* rare-symbol recall.

## Jalur utama 2: ConvNeXt-CTC

Peran:

* kandidat CTC dengan visual encoder paling kuat;
* membandingkan keuntungan pretraining ImageNet;
* kandidat ensemble dengan TrOCR.

Jangan berasumsi ConvNeXt pasti unggul. Jika horizontal feature map terlalu terkompresi, ResNet custom dapat lebih baik.

## Jalur utama 3: TrOCR-Small aspect-ratio-aware

Peran:

* kandidat autoregressive utama;
* memiliki language prior;
* membantu mengenali urutan kata;
* realistis untuk dataset sekitar 4.100 gambar.

Ini harus lebih diprioritaskan daripada TrOCR default.

## Jalur utama 4: PyLaia atau Kraken

Peran:

* menguji keuntungan pretraining historical handwriting;
* mengenali gaya huruf historis;
* memberikan error pattern berbeda dari ImageNet dan IAM-based models.

Pilih satu checkpoint yang paling dekat berdasarkan:

* periode dokumen;
* bahasa;
* bentuk alfabet;
* handwriting style;
* transcription convention;
* charset.

## Model sekunder

### CRNN awal

Gunakan untuk:

* sanity check;
* overfit test;
* baseline murah.

### TrOCR-Base

Gunakan setelah TrOCR-Small sudah stabil dan preprocessing terbaik sudah ditemukan.

Gunakan:

* partial freezing;
* learning rate kecil;
* gradual unfreezing.

### TrOCR-Large

Prioritas rendah. Kapasitas besar tidak menjamin hasil lebih baik pada dataset kecil.

### PaddleOCR

Gunakan sebagai:

* model cadangan;
* diversity model;
* eksperimen jika model utama telah stabil.

---

# 9. Strategi training

## Tahap sanity check

Setiap keluarga utama harus mampu meng-overfit 32–100 sampel tanpa augmentasi.

Model harus mencapai training CER yang sangat rendah.

Jika gagal, periksa:

* vocabulary;
* tokenizer round-trip;
* target truncation;
* CTC alignment;
* generation length;
* image clipping;
* padding mask;
* decoding;
* evaluasi.

Jangan melakukan tuning besar sebelum tes ini berhasil.

## Validation split

Gunakan split yang konsisten untuk semua model.

Stratifikasi, sejauh memungkinkan, berdasarkan:

* panjang target;
* aspect ratio;
* keberadaan simbol langka;
* multiple spaces;
* kualitas gambar;
* writer identity jika tersedia.

Gunakan satu fold untuk iterasi cepat, kemudian konfirmasi model terbaik pada fold lain.

## Optimizer dan learning rate

Untuk model custom:

* mulai dengan AdamW;
* gunakan scheduler;
* gunakan gradient clipping;
* pilih checkpoint berdasarkan validation CER atau score.

Untuk pretrained model:

* learning rate encoder lebih kecil;
* learning rate decoder atau head dapat lebih besar;
* partial freezing pada fase awal;
* gradual unfreezing bila diperlukan.

## Batch construction

Gunakan width bucketing agar gambar dengan panjang serupa berada dalam batch yang sama.

Manfaatnya:

* mengurangi padding;
* memperbesar effective batch size;
* mempercepat training;
* mengurangi perbedaan panjang sequence ekstrem dalam satu batch.

## Early stopping

Pantau:

* validation CER;
* validation WER;
* score;
* rare-symbol recall;
* long-line CER.

Jangan menghentikan training hanya berdasarkan validation loss.

---

# 10. Strategi CTC

CTC kemungkinan menjadi jalur paling literal untuk kompetisi ini.

## Hal yang wajib diperiksa

* blank index;
* input length;
* target length;
* repeated characters;
* repeated spaces;
* invalid alignment;
* blank dominance;
* horizontal stride;
* `zero_infinity`.

Jangan membiarkan `zero_infinity=True` menyembunyikan sampel invalid tanpa logging.

## Decoding CTC

Bandingkan:

1. greedy decoding;
2. beam search tanpa language model;
3. beam search dengan character-level LM ringan.

Karena label mengandung:

* ejaan historis;
* simbol;
* multiple spaces;
* fragmen kalimat;

maka **character-level LM lebih aman daripada word-level modern LM**.

---

# 11. Strategi TrOCR

TrOCR sudah memiliki komponen contextual language modeling melalui autoregressive decoder.

## Yang wajib diaudit

* target tidak terpotong;
* tokenizer mempertahankan simbol;
* tokenizer mempertahankan multiple spaces;
* `clean_up_tokenization_spaces=False`;
* generation length cukup;
* EOS tidak muncul terlalu dini;
* padding label dimask dengan benar;
* special token tidak masuk output.

## Decoding awal

Gunakan baseline konservatif:

* greedy;
* beam size 2–5;
* tanpa repetition penalty;
* tanpa `no_repeat_ngram_size`;
* length penalty mendekati netral.

Beam yang terlalu besar dapat membuat model menghasilkan kata lebih masuk akal, tetapi kurang literal.

---

# 12. Language model dan contextual decoding

Gagasan Anda benar, tetapi yang dibutuhkan adalah **contextual language prior**, bukan reasoning bebas.

## Fungsi language model

LM membantu ketika visual ambigu, misalnya memilih antara beberapa kandidat karakter atau kata.

Namun LM tidak boleh:

* memodernisasi ejaan;
* memperbaiki grammar;
* menghilangkan simbol;
* menebak teks pada kertas yang hilang;
* mengganti fragmen dengan kalimat lengkap.

## Rekomendasi

Untuk CTC:

* gunakan character n-gram LM;
* latih dari transkripsi train;
* tambahkan external historical corpus hanya jika aturan kompetisi mengizinkan;
* gunakan shallow fusion dengan bobot kecil;
* tuning LM weight pada validation.

Untuk TrOCR:

* jangan menambahkan LLM eksternal sebagai korektor utama;
* gunakan beam reranking secara konservatif;
* biarkan visual evidence tetap dominan.

## Multiple spaces

LM tidak boleh menjadi penentu utama jumlah spasi. Multiple spaces pada dataset dapat merepresentasikan gap fisik, bukan struktur linguistik.

Informasi tersebut lebih tepat dipelajari dari:

* panjang visual gap;
* posisi horizontal;
* alignment model.

---

# 13. Error analysis

Setiap eksperimen harus menghasilkan laporan lebih dari sekadar CER dan WER.

## Kelompok error

* insertion;
* deletion;
* substitution;
* spacing error;
* case error;
* punctuation error;
* rare-symbol error;
* truncated ending;
* repeated-character error;
* long-line error;
* damaged-region error.

## Evaluasi per kelompok data

Hitung metrik untuk:

* line pendek;
* line sedang;
* line panjang;
* aspect ratio ekstrem;
* line dengan `^`;
* line dengan `*`;
* line dengan multiple spaces;
* line dengan kerusakan fisik;
* line berkualitas rendah.

## Confusion table

Contoh yang perlu dianalisis:

```text
^  → hilang
*  → .
l  → i
rn → m
c  → e
s  → f
```

Dari hasil tersebut baru ditentukan apakah eksperimen berikutnya harus mengubah:

* resolusi;
* sampling;
* augmentation;
* decoder;
* language model;
* checkpoint.

---

# 14. Post-training

## Checkpoint averaging

Untuk model yang stabil, averaging beberapa checkpoint terbaik dapat mengurangi fluktuasi.

Contohnya:

* tiga checkpoint CER terbaik;
* checkpoint dari akhir training yang saling berdekatan.

## Multi-fold training

Setelah konfigurasi terbaik ditemukan:

* train pada beberapa fold;
* simpan out-of-fold prediction;
* hitung mean dan variance;
* ensemble fold model.

Ini lebih bernilai daripada menambah arsitektur baru.

## Ensemble model

Kombinasi paling potensial:

1. ConvNeXt-CTC + TrOCR-Small aspect-aware;
2. ResNet-CTC + TrOCR;
3. TrOCR + Kraken/PyLaia;
4. CTC + historical pretrained HTR.

Alasannya:

* CTC lebih literal;
* TrOCR lebih kontekstual;
* Kraken/PyLaia memiliki prior bentuk tulisan historis.

Jangan hanya menggabungkan model dengan skor tertinggi. Pilih model dengan error yang berbeda.

Dengan ensemble CSV yang sudah tersedia, gunakan strategi yang divalidasi pada OOF atau validation:

* exact-string majority vote;
* pilih model utama ketika tidak ada kesepakatan;
* gunakan confidence bila tersedia;
* jangan membuat aturan manual berdasarkan test prediction.

## Post-processing

Post-processing hanya boleh bersifat teknis:

* menghapus special token;
* mengembalikan encoding Unicode;
* mempertahankan multiple spaces;
* menghapus newline yang tidak sah;
* memastikan format CSV benar.

Hindari:

* spell correction;
* grammar correction;
* normalisasi ejaan modern;
* penggantian kata berdasarkan dictionary;
* collapse multiple spaces tanpa mengetahui evaluator.

---

# 15. Urutan eksperimen yang direkomendasikan

## Fase 1: Audit

1. Samakan evaluator.
2. Audit Unicode dan whitespace.
3. Audit charset dan tokenizer.
4. Audit target truncation.
5. Audit image clipping.
6. Audit CTC alignment.
7. Jalankan small-subset overfit test.

## Fase 2: Geometri input

1. Uji target height 64, 96, dan 128.
2. Uji max-width atau canvas-width.
3. Ukur pixel-per-character.
4. Terapkan width bucketing.
5. Pastikan internal gap dipertahankan.

## Fase 3: Model utama

1. Stabilkan ResNet-CTC.
2. Stabilkan ConvNeXt-CTC.
3. Stabilkan TrOCR-Small aspect-aware.
4. Bandingkan berdasarkan error category.

## Fase 4: Data engineering

1. Rare-symbol sampling.
2. Long-line sampling.
3. Augmentasi ringan.
4. Audit label anomali.
5. Evaluasi precision dan recall simbol.

## Fase 5: Decoding

1. CTC greedy.
2. CTC beam tanpa LM.
3. CTC beam dengan character LM ringan.
4. TrOCR greedy.
5. TrOCR beam kecil.

## Fase 6: Historical transfer

1. Pilih satu PyLaia checkpoint.
2. Pilih satu Kraken checkpoint.
3. Audit charset dan transcription convention.
4. Fine-tune.
5. Bandingkan terutama pada handwriting historis dan simbol.

## Fase 7: Konfirmasi

1. Jalankan konfigurasi terbaik pada fold lain.
2. Lakukan checkpoint averaging.
3. Buat out-of-fold prediction.
4. Ukur kestabilan skor.

## Fase 8: Ensemble

1. Ensemble CTC terbaik dengan TrOCR terbaik.
2. Tambahkan PyLaia/Kraken bila memberikan error diversity.
3. Validasi strategi pada OOF.
4. Buat submission final.

---

# Rekomendasi sistem final

Sistem akhir yang paling masuk akal adalah:

### Model literal utama

**ConvNeXt-CTC atau ResNet-CTC**

Fungsi:

* mempertahankan karakter;
* mempertahankan simbol;
* lebih kecil risiko hallucination;
* menangani transkripsi literal.

### Model kontekstual utama

**TrOCR-Small aspect-ratio-aware**

Fungsi:

* membantu pengenalan kata;
* menangani bentuk huruf ambigu;
* meningkatkan struktur sequence;
* menurunkan WER.

### Model historical specialist

**PyLaia atau Kraken dengan checkpoint paling dekat**

Fungsi:

* memberikan representasi handwriting historis;
* meningkatkan pengenalan bentuk huruf kuno;
* menyediakan diversity untuk ensemble.

### Decoder tambahan

**Character-level LM ringan untuk model CTC**

Fungsi:

* melakukan reranking kandidat;
* membantu urutan karakter;
* tetap mempertahankan ejaan historis.

### Final output

**Ensemble model literal + contextual + historical specialist**, kemudian hanya menjalankan post-processing teknis yang tidak mengubah isi transkripsi.

# Konklusi akhir

Untuk kompetisi ini, hasil terbaik kemungkinan tidak datang dari satu model besar yang “memahami bahasa”, melainkan dari kombinasi:

[
\text{visual fidelity}
+
\text{historical handwriting knowledge}
+
\text{controlled language context}
+
\text{exact label handling}
]

Fokus utama Anda seharusnya:

1. mempertahankan geometri horizontal gambar;
2. mempertahankan seluruh karakter dan whitespace;
3. menggunakan CTC sebagai recognizer literal;
4. menggunakan TrOCR sebagai recognizer kontekstual;
5. menggunakan PyLaia/Kraken sebagai historical specialist;
6. menggunakan language model hanya sebagai reranker terbatas;
7. melakukan error analysis per karakter dan per jenis line;
8. melakukan ensemble setelah model individual benar-benar stabil.

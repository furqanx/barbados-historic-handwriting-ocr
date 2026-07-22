Menurut saya, **jangan langsung berpindah-pindah model atau melakukan random hyperparameter search**. CRNN+CTC dan TrOCR seharusnya sudah cukup kuat untuk dataset Anda. Bila keduanya sama-sama belum menghasilkan CER/WER yang memuaskan, kemungkinan hambatan utamanya berada pada **pipeline data, representasi label, resolusi input, alignment, decoding, atau evaluasi**, bukan kekurangan arsitektur.

Pretrained HTR tetap layak menjadi eksperimen berikutnya, tetapi model baru tidak otomatis menyelesaikan masalah karakter seperti `^`, `*`, tanda baca, atau simbol langka.

## Hal pertama yang perlu dibuktikan

Lakukan eksperimen overfit pada subset sangat kecil, misalnya **32–100 gambar**.

Targetnya bukan generalisasi, tetapi memastikan model mampu mencapai:

* training CER mendekati 0;
* training WER mendekati 0;
* prediksi simbol `^`, `*`, dan tanda baca secara tepat;
* panjang output yang benar.

Bila CRNN atau TrOCR **tidak dapat menghafal 32–100 contoh**, jangan melakukan hyperparameter tuning skala besar. Hampir pasti ada masalah seperti:

* label tidak identik dengan vocabulary;
* karakter berubah atau hilang saat preprocessing;
* target terpotong;
* konfigurasi blank/padding salah;
* panjang sequence CTC tidak mencukupi;
* decoder TrOCR berhenti terlalu awal;
* implementasi CER/WER tidak konsisten dengan evaluator kompetisi.

Ini merupakan tes diagnostik paling bernilai sebelum eksperimen lanjutan.

---

## Karakter `^`, `*`, dan simbol langka bukan terutama masalah model

Model hanya akan mengenali simbol dengan baik bila tiga hal terpenuhi:

1. Simbol benar-benar ada dalam output vocabulary.
2. Simbol muncul cukup sering dalam training data.
3. Simbol dipertahankan tanpa perubahan dalam preprocessing label dan evaluasi.

Buat statistik frekuensi setiap karakter. Misalnya:

```text
a     24.831
e     20.412
space 18.920
^         37
*         19
```

Bila `*` hanya muncul 19 kali, mengganti ConvNeXt dengan backbone lain hampir tidak akan menyelesaikannya. Model memang tidak mendapatkan cukup contoh visual.

Dalam kondisi tersebut, yang lebih masuk akal adalah:

* oversampling line yang mengandung simbol langka;
* weighted sampling berdasarkan rarity karakter;
* memastikan augmentasi tidak menghilangkan goresan kecil;
* memeriksa manual seluruh sampel yang mengandung simbol tersebut;
* bila memungkinkan, menambahkan data sintetis atau pseudo-label berkualitas tinggi.

Saya tidak menyarankan langsung memberikan class weight ekstrem pada CTC karena efeknya terhadap alignment dapat sulit dikendalikan. **Sampling berbasis line** biasanya lebih mudah dianalisis.

### Audit Unicode

Pastikan train label, tokenizer, prediksi, dan evaluator menggunakan bentuk Unicode yang sama. Unicode memiliki beberapa normalization form seperti NFC, NFD, NFKC, dan NFKD; karakter yang tampak sama dapat terdiri dari code point berbeda. ([Unicode][1])

Untuk kompetisi, gunakan satu normalisasi yang konsisten—biasanya NFC—tetapi hanya bila sesuai dengan evaluator. Hindari NFKC/NFKD tanpa alasan karena compatibility normalization dapat mengubah representasi karakter tertentu.

Uji seluruh vocabulary dengan round-trip:

```python
decoded = tokenizer.decode(tokenizer.encode(text))
assert decoded == text
```

Lakukan terhadap **setiap target**, bukan hanya beberapa contoh.

---

## Untuk CRNN + CTC, audit alignment sebelum tuning

CTC memerlukan jumlah time step output encoder yang cukup untuk merepresentasikan target. Secara dasar, target length tidak boleh melebihi input sequence length. ([PyTorch Documentation][2])

Namun pemeriksaan praktisnya harus lebih ketat. Karakter berulang seperti:

```text
letter
committee
***
```

membutuhkan separation melalui blank pada alignment CTC. Karena itu, minimum time step efektif bukan sekadar jumlah karakter, tetapi kira-kira:

[
T_{\min}
========

U
+
\text{jumlah pasangan karakter berulang bersebelahan}
]

dengan (U) panjang target.

Untuk line maksimum sekitar 120 karakter, horizontal stride yang terlalu besar dapat membuat alignment mustahil. Contohnya:

```text
input width 768
horizontal stride 8
T = 96
target length = 120
```

Sampel tersebut tidak mungkin dipelajari dengan benar.

Periksa untuk setiap sampel:

```text
encoder_time_steps
target_length
adjacent_repeat_count
required_ctc_steps
```

Jangan hanya mengaktifkan `zero_infinity=True` lalu melanjutkan training. Opsi tersebut dapat menyembunyikan sampel invalid dengan membuat loss-nya nol. Catat jumlah sampel invalid pada setiap batch.

Hal lain yang perlu diverifikasi adalah `input_lengths`. Jika gambar dipad ke lebar yang sama, CTC seharusnya menerima **panjang sequence asli setelah encoder**, bukan selalu panjang padding maksimum batch.

### Loss CTC bukan CER

CTC loss dan CER tidak harus bergerak secara paralel. Loss dapat menurun karena distribusi probabilitas alignment membaik, sementara hasil greedy decoding belum berubah. Karena itu:

* gunakan loss untuk mendeteksi kestabilan optimisasi;
* pilih checkpoint berdasarkan validation CER atau competition score;
* jangan memilih checkpoint hanya karena validation loss paling rendah.

CTC juga sangat dipengaruhi decoder. Greedy decoding bagus sebagai baseline, tetapi beam search dapat menghasilkan transkripsi berbeda dan dapat digabung dengan character language model. Penelitian pada HTR menunjukkan bahwa decoding yang lebih kuat dapat mengungguli best-path decoding. ([Repositum][3])

Namun language model perlu digunakan hati-hati. LM yang terlalu kuat cenderung “memperbaiki” kata menjadi lebih umum dan dapat menghapus simbol seperti `^` atau `*`. Untuk target yang mengandung simbol nonlinguistik, bandingkan:

```text
greedy
beam tanpa LM
beam + character LM ringan
```

Jangan menganggap beam atau LM pasti lebih baik.

---

## Untuk TrOCR, masalah terbesar biasanya tokenizer dan decoding

TrOCR dilatih menggunakan cross-entropy dalam kondisi teacher forcing, tetapi dievaluasi melalui autoregressive generation. Jadi training loss yang baik tidak menjamin generation CER yang baik.

Periksa khususnya:

### Target truncation

Hitung panjang target **setelah tokenization**, bukan panjang karakter asli.

Target 120 karakter bisa menghasilkan jumlah token berbeda tergantung tokenizer. Pastikan:

```text
max_target_token_length
max_length / max_new_tokens
EOS behavior
padding token
decoder start token
```

Tidak ada target yang terpotong. Hugging Face membatasi panjang generation melalui `max_length` atau `max_new_tokens`; konfigurasi yang terlalu pendek dapat memotong bagian akhir line. ([Hugging Face][4])

### Jangan gunakan pembatas generasi NLP secara sembarangan

Konfigurasi seperti ini dapat merusak HTR:

```text
no_repeat_ngram_size > 0
repetition_penalty tinggi
length_penalty agresif
early stopping agresif
```

Tulisan historis secara sah dapat memiliki karakter, suku kata, atau kata berulang. Untuk baseline, gunakan decoding yang sedekat mungkin dengan probabilitas visual:

```text
greedy decoding
beam kecil, misalnya 2–5
tanpa repetition penalty
tanpa no-repeat constraint
```

Beam besar belum tentu menurunkan CER. Beam besar dapat meningkatkan fluency, tetapi juga meningkatkan substitusi linguistik dan menghilangkan simbol aneh.

### Tokenizer bawaan versus target karakter

Periksa bagaimana tokenizer merepresentasikan:

```text
^
*
**
^^
word*
^word
spasi berulang
tanda kutip historis
dash yang berbeda
```

Yang penting bukan apakah simbol “tersedia”, tetapi apakah encoding dan decoding-nya stabil. TrOCR bisa saja secara visual melihat `*`, tetapi decoder lebih menyukai token kata umum karena prior bahasa dari pretrained decoder.

Bila kompetisi sangat character-sensitive, CRNN-CTC dengan vocabulary karakter tertutup terkadang lebih literal daripada TrOCR.

---

## Lakukan error analysis, bukan hanya membaca satu angka CER

Ambil sekitar 100–200 prediksi validation terburuk dan kelompokkan error menjadi:

* substitution karakter visual;
* deletion;
* insertion;
* kehilangan akhir kalimat;
* spasi salah;
* huruf kapital;
* simbol langka;
* karakter berulang;
* line sangat panjang;
* gambar buram atau kontras rendah;
* kesalahan anotasi.

Dari distribusi ini, keputusan eksperimen menjadi jelas.

Contohnya:

| Pola error dominan                       | Kemungkinan perbaikan                                              |
| ---------------------------------------- | ------------------------------------------------------------------ |
| Akhir line sering hilang                 | Decoder max length, EOS, input terlalu dipadatkan                  |
| Karakter berulang hilang pada CTC        | Time step kurang atau decoding salah                               |
| `^`, `*`, tanda baca sering hilang       | Data langka, resizing, augmentasi terlalu kuat, LM terlalu dominan |
| Kata terlihat benar tetapi spacing salah | Label normalization dan decoder                                    |
| Training bagus, validation buruk         | Overfitting, split, writer/domain shift                            |
| Training juga buruk                      | Pipeline, alignment, resolusi, learning rate, vocabulary           |
| Loss membaik tetapi CER stagnan          | Objective–decoding mismatch                                        |

Error analysis biasanya lebih produktif daripada mencoba sepuluh optimizer atau lima backbone.

---

## Kapan hyperparameter tuning dilakukan?

Setelah semua audit di atas lolos.

Hyperparameter yang paling bernilai untuk HTR Anda bukan semuanya sekaligus, melainkan:

1. **Input height dan horizontal resolution**
   Misalnya 64, 96, dan 128. Simbol kecil seperti `*` dan `^` dapat hilang bila height terlalu rendah.

2. **Horizontal stride encoder**
   Terutama untuk CTC dan target panjang.

3. **Learning rate dan scheduler**
   Ini tetap penting, tetapi dilakukan setelah alignment valid.

4. **Augmentation strength**
   Blur, erosion, perspective, dan resize yang terlalu kuat dapat menghilangkan punctuation.

5. **Regularisasi dan freezing**
   Terutama untuk TrOCR/pretrained HTR pada dataset kecil.

6. **Decoding configuration**
   Greedy, beam width, character LM weight, dan length normalization.

Gunakan **controlled ablation**, bukan mengganti beberapa hal sekaligus. Contohnya, pertahankan seluruh konfigurasi dan hanya ubah input height. Setelah mendapatkan height terbaik, baru uji learning rate.

---

## Pendapat saya mengenai pretrained OCR/HTR berikutnya

Itu adalah langkah yang masuk akal, tetapi pilih checkpoint berdasarkan:

* kesamaan gaya tulisan;
* periode dokumen;
* bahasa dan alfabet;
* aturan transkripsi;
* keberadaan simbol;
* line height dan preprocessing.

Model historical HTR yang domainnya mirip kemungkinan lebih berguna daripada pretrained model yang secara umum lebih besar.

Namun sebelum masuk ke sana, saya akan memastikan minimal satu dari CRNN atau TrOCR dapat:

1. meng-overfit subset kecil;
2. mempertahankan seluruh karakter secara lossless;
3. tidak memiliki target truncation;
4. memiliki alignment CTC yang valid;
5. menunjukkan error profile yang jelas.

**Kesimpulan saya:** jangan menjadikan pergantian hyperparameter atau model sebagai langkah pertama. Prioritas tertinggi sekarang adalah membuktikan bahwa pipeline memiliki kemampuan mengenali semua karakter secara sempurna pada subset kecil, lalu menemukan kategori error yang mendominasi validation. Setelah itu barulah tuning terarah dan transfer learning dari pretrained historical HTR akan memberi hasil yang dapat dipercaya.

[1]: https://www.unicode.org/reports/tr15/?utm_source=chatgpt.com "UAX #15: Unicode Normalization Forms"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html?utm_source=chatgpt.com "CTCLoss — PyTorch 2.13 documentation"
[3]: https://repositum.tuwien.at/retrieve/1835?utm_source=chatgpt.com "A Connectionist Temporal Classification Decoding Algorithm"
[4]: https://huggingface.co/docs/transformers/v4.15.0/main_classes/model?utm_source=chatgpt.com "Models · Hugging Face"

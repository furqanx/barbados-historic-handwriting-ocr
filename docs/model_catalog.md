# Model Catalog

Dokumen ini merangkum pendekatan model yang pernah dipertimbangkan untuk kompetisi R.O.A.D. Barbados Historic Handwriting Challenge. Gunakan ini sebagai katalog strategi, bukan sebagai runbook eksekusi. Command eksekusi ada di `cloud_training.md` dan workflow-specific docs.

## Kesimpulan utama

Untuk dataset hanya **±4.100 line images**, urutan eksperimen paling rasional adalah:

1. **CRNN + CTC yang aspect-ratio-aware** sebagai baseline kuat.
2. **TrOCR-Small-Handwritten** sebagai pretrained Transformer paling realistis.
3. **TrOCR-Base-Handwritten** sebagai kandidat utama bila GPU cukup.
4. **PyLaia/Kraken pretrained historical HTR** bila bahasa, aksara, periode, dan gaya transkripsinya cukup dekat.
5. **Donut** hanya sebagai eksperimen tambahan, bukan jalur utama.

Alasan terbesarnya adalah bentuk gambar. Dengan aspect ratio median **16,8**, resize input harus mempertahankan rasio. Preprocessor bawaan TrOCR-Base menggunakan ukuran `384`, yang secara default menghasilkan input persegi; penggunaan langsung berisiko memampatkan garis tulisan yang sangat panjang. ([Hugging Face][1])

---

## 1. CRNN + CTC

| Model/arsitektur                        | Sumber/library/checkpoint                                                                    |   Cocok untuk handwritten line recognition | Kelebihan                                                                                                                                                                          | Kekurangan                                                                                                                        | Compute                                        | Kesulitan         | Prioritas                      |
| --------------------------------------- | -------------------------------------------------------------------------------------------- | -----------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------- | ------------------------------ |
| **CNN + BiLSTM + CTC custom**           | PyTorch, `torch.nn.CTCLoss`                                                                  |                           **Sangat cocok** | Mempertahankan urutan horizontal; menerima variable-width image; tokenizer karakter sederhana; cepat; mudah mengontrol jumlah time step                                            | Tidak memiliki visual/language pretraining; WER dapat tertinggal dari model autoregresif; perlu decoding dan augmentasi yang baik | **Rendah**. Umumnya realistis pada GPU 8–12 GB | **Sedang**        | **P0, wajib**                  |
| **ResNet/ConvNeXt-tiny + BiLSTM + CTC** | PyTorch, `timm`                                                                              |                           **Sangat cocok** | Encoder lebih kuat daripada CNN sederhana; masih efisien; dapat memakai ImageNet initialization                                                                                    | Downsampling horizontal harus diawasi; backbone standar sering terlalu banyak mengecilkan lebar                                   | **Rendah–sedang**. Sekitar 8–16 GB             | **Sedang**        | **P0–P1**                      |
| **PyLaia CRNN-CTC**                     | PyLaia/Teklia; checkpoint seperti `Teklia/pylaia-iam`, `pylaia-himanis`, `pylaia-norhand-v1` |                           **Sangat cocok** | Dibuat khusus untuk ATR/HTR; mendukung training, inference, CTC decoding, dan character language model; model resminya meresize ke fixed height sambil mempertahankan aspect ratio | Ekosistem lebih khusus daripada PyTorch/HF; format charset dan konfigurasi perlu dipelajari                                       | **Rendah–sedang**                              | **Sedang**        | **P1 tinggi**                  |
| **Kraken recognition network / VGSL**   | Kraken, eScriptorium, model repository melalui `kraken list`                                 | **Sangat cocok**, khususnya historical HTR | Memang dioptimalkan untuk historical dan non-Latin text; variable architecture; banyak model historis siap fine-tune                                                               | Workflow dan format data lebih khusus; checkpoint sangat sensitif terhadap bahasa, skrip, dan transcription convention            | **Rendah–sedang**                              | **Sedang–tinggi** | **P1**, bila checkpoint sesuai |
| **Calamari CNN-LSTM-CTC ensemble**      | `Calamari-OCR/calamari` dan pretrained mixed models                                          |                                  **Cocok** | Line-based; mudah membuat ensemble beberapa recognizer; relatif ringan                                                                                                             | Ekosistem lebih lama; integrasi eksperimen modern tidak senyaman PyTorch/HF; checkpoint belum tentu sesuai domain                 | **Rendah**                                     | **Sedang**        | **P2**                         |

PyLaia secara eksplisit ditujukan untuk Automatic Text Recognition dengan jaringan convolutional-recurrent. Checkpoint resminya menggunakan fixed height 128 sambil mempertahankan aspect ratio, tepat untuk gambar line yang sangat lebar. ([Teklia Documentation][2])

Kraken juga secara khusus menyatakan fokus pada historical dan non-Latin text recognition serta menyediakan repositori model publik. ([Kraken][3])

### Konfigurasi CRNN yang saya rekomendasikan

```text
Input RGB
→ resize height 64 atau 96, preserve aspect ratio
→ pad width per batch
→ CNN/ResNet encoder
→ feature map dengan tinggi akhir 1
→ sequence sepanjang dimensi width
→ 2–3 layer BiLSTM
→ linear classifier
→ CTC loss
```

Hal terpenting adalah memastikan:

[
T_{\text{encoder}} \geq \text{panjang target CTC}
]

Untuk target maksimum sekitar 120 karakter, hindari horizontal downsampling terlalu besar. Saya akan memakai total horizontal stride sekitar **4–8**, bukan langsung stride 16 atau 32.

### Kandidat baseline praktis

* Height: `64`, `96`, atau `128`.
* Width: dynamic padding dengan batas misalnya `1536–2048`.
* Charset: karakter aktual dari seluruh training set, bukan vocabulary generik.
* Decoder pertama: greedy CTC.
* Decoder lanjutan: beam search + character n-gram LM.
* Loss: CTC standar, kemudian bandingkan focal-CTC hanya bila terdapat masalah imbalance karakter.

PyLaia menunjukkan bahwa character 6-gram language model dapat menurunkan CER dan WER pada beberapa dataset HTR, walaupun besarnya keuntungan sangat bergantung pada kesamaan distribusi teks. ([Hugging Face][4])

---

## 2. TrOCR / Vision Encoder–Decoder

| Model/checkpoint                                   | Sumber                                                         |       Cocok untuk handwritten line recognition | Kelebihan                                                                                                                                                  | Kekurangan                                                                                                                                                       | Compute                                                           | Kesulitan         | Prioritas     |
| -------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------- | ------------- |
| **TrOCR-Small-Handwritten**                        | `microsoft/trocr-small-handwritten`, Hugging Face Transformers |                   **Sangat cocok secara task** | Fine-tuned pada IAM handwriting; hanya sekitar 62M parameter; autoregressive decoder membantu word-level consistency; paling realistis untuk dataset kecil | Default preprocessing tidak ideal untuk line sangat lebar; tokenizer generatif dapat hallucinate, menghapus, atau mengganti kata; decoding lebih lambat dari CTC | **Sedang**. Sekitar 8–16 GB dengan FP16 dan gradient accumulation | **Rendah–sedang** | **P0–P1**     |
| **TrOCR-Base-Handwritten**                         | `microsoft/trocr-base-handwritten`                             |                   **Sangat cocok secara task** | Pretrained image dan text Transformer; sekitar 334M parameter; umumnya kandidat accuracy terbaik yang masih realistis                                      | Lebih mudah overfit; kebutuhan VRAM tinggi; square resizing dan positional embedding perlu ditangani; inference autoregresif                                     | **Sedang–tinggi**. Idealnya 16–24 GB                              | **Sedang**        | **P1 tinggi** |
| **TrOCR-Large-Handwritten**                        | `microsoft/trocr-large-handwritten`                            |             **Cocok**, tetapi kurang realistis | Kapasitas terbesar; model asli melaporkan CER IAM lebih baik daripada small/base                                                                           | Sekitar 558M parameter; mahal; keuntungan mungkin kecil pada 4.1k data; tuning dan cross-validation lebih lambat                                                 | **Tinggi**. Sekitar 24–40 GB atau lebih                           | **Sedang–tinggi** | **P2–P3**     |
| **TrOCR-Small dengan aspect-ratio padding custom** | Small checkpoint + custom image processor                      | **Pilihan TrOCR yang paling direkomendasikan** | Menghindari pemampatan gambar; compute masih terkendali; cocok untuk ablation cepat                                                                        | Positional embedding encoder mungkin perlu interpolasi; batching variable width lebih rumit                                                                      | **Sedang**                                                        | **Sedang–tinggi** | **P0–P1**     |
| **TrOCR-Base dengan partial freezing**             | Base checkpoint; freeze sebagian encoder/decoder               |              **Sangat cocok untuk data kecil** | Mengurangi overfitting dan VRAM aktivasi; memungkinkan gradual unfreezing                                                                                  | Terlalu banyak freezing dapat menghambat adaptasi terhadap tulisan historis                                                                                      | **Sedang**                                                        | **Sedang**        | **P1**        |

Checkpoint Microsoft memang fine-tuned pada IAM dan ditujukan untuk OCR pada single text-line images. TrOCR-Small memiliki sekitar 62M parameter, Base 334M, dan Large 558M. Pada eksperimen resmi IAM, cased CER yang dilaporkan adalah 4,22 untuk Small, 3,42 untuk Base, dan 2,89 untuk Large. Angka tersebut bukan perkiraan skor dataset Anda, tetapi menunjukkan trade-off kapasitas model. ([Hugging Face][5])

### Masalah terbesar TrOCR untuk dataset Anda

Checkpoint bawaan menggunakan image size 384. Untuk line dengan aspect ratio 16,8, resize langsung ke `384 × 384` akan mengubah satu karakter yang semula relatif lebar menjadi sangat sempit. ([Hugging Face][1])

Lebih aman membandingkan setidaknya dua preprocessing:

#### A. Default TrOCR

```text
original line → resize 384 × 384
```

Gunakan hanya sebagai baseline karena mudah diimplementasikan.

#### B. Preserve-ratio canvas

```text
original line
→ resize ke fixed height, misalnya 96 atau 128
→ pertahankan aspect ratio
→ pad ke canvas lebar, misalnya 1024, 1536, atau 2048
→ patch embedding
```

Namun canvas terlalu lebar meningkatkan jumlah visual tokens dan konsumsi memori. Alternatif yang lebih aman adalah memakai:

* width bucketing;
* random horizontal crops yang tetap menyimpan target konsisten, bila dapat dilakukan;
* overlapping-window inference untuk line ekstrem;
* interpolasi positional embeddings;
* batch size kecil dengan gradient accumulation.

### Urutan eksperimen TrOCR

1. `trocr-small-handwritten`, preprocessing default.
2. Small dengan preserve-ratio padding.
3. Small dengan augmentasi historis.
4. `trocr-base-handwritten`, encoder sebagian dibekukan.
5. Base full fine-tuning dengan learning rate lebih rendah.
6. Large hanya jika Base sudah jelas unggul dan compute memungkinkan.

Untuk Small, coba learning rate sekitar `1e-5` sampai `5e-5`. Untuk Base, mulai lebih konservatif, misalnya `5e-6` sampai `2e-5`.

---

## 3. Donut / OCR-free document model

| Model/checkpoint                                          | Sumber                                                        |             Cocok untuk handwritten line recognition | Kelebihan                                                                                           | Kekurangan                                                                                                                                                            | Compute                                  | Kesulitan         | Prioritas       |
| --------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------: | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ----------------- | --------------- |
| **Donut-Base**                                            | `naver-clova-ix/donut-base`, Hugging Face Transformers        | **Secara teknis bisa, tetapi bukan target utamanya** | Swin encoder menerima visual document; BART decoder dapat menghasilkan sequence panjang; end-to-end | Pretraining berorientasi document understanding, bukan historical line transcription; model besar; tokenizer dan task prompt perlu disesuaikan; risiko overfit tinggi | **Tinggi**. Biasanya 16–24 GB atau lebih | **Tinggi**        | **P3**          |
| **Donut-Base fine-tuned sebagai transcription generator** | Donut-Base + custom task token, misalnya `<s_transcribe>`     |                         **Cukup sebagai eksperimen** | Dapat membuat satu task transcription sederhana; decoder memiliki language prior                    | Memerlukan penyesuaian data collator, prompt, tokenizer, max length, dan decoding; tidak memberi keuntungan jelas atas TrOCR                                          | **Tinggi**                               | **Tinggi**        | **P3**          |
| **Donut-CORD/DocVQA checkpoint**                          | `donut-base-finetuned-cord-v2`, `donut-base-finetuned-docvqa` |                           **Tidak direkomendasikan** | Sudah fine-tuned untuk image-to-structured-text                                                     | Prior output berupa JSON, field extraction, atau jawaban; domain receipts/documents sangat berbeda dari historical handwriting                                        | **Tinggi**                               | **Sedang–tinggi** | **P4, hindari** |

Donut terdiri dari **Swin visual encoder dan BART text decoder**. Model ini diperkenalkan untuk visual document understanding seperti document parsing, classification, information extraction, dan DocVQA, bukan sebagai recognizer khusus single handwritten line. ([GitHub][6])

Checkpoint CORD dan DocVQA membawa prior task yang tidak relevan untuk transkripsi polos. Repository resmi Donut sendiri menonjolkan document parsing, ticket parsing, document classification, dan visual question answering. ([GitHub][6])

### Penilaian untuk kompetisi ini

Donut bukan model yang buruk, tetapi **alokasi compute-to-probability-of-improvement** kurang menarik:

* data terlalu kecil untuk adaptasi besar;
* tidak ada kebutuhan memahami layout karena input sudah berupa satu baris;
* keunggulan Donut dalam document-level reasoning tidak terpakai;
* TrOCR memberikan inductive bias yang lebih dekat dengan task;
* CRNN/PyLaia memberikan representasi horizontal yang lebih natural.

Saya hanya akan menjalankan Donut setelah CRNN dan TrOCR sudah stabil, sebagai kandidat ensemble yang menghasilkan error berbeda.

---

## 4. Pretrained OCR/HTR models dari Hugging Face atau library lain

Bagian ini memang sedikit tumpang tindih dengan CRNN karena banyak model HTR historis pretrained memakai CNN-RNN-CTC.

| Model/checkpoint                                     | Sumber                                                 |                          Cocok untuk historical handwritten lines | Kelebihan                                                                                                         | Kekurangan                                                                                                                                       | Compute           | Kesulitan         | Prioritas                            |
| ---------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------: | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ----------------- | ------------------------------------ |
| **Teklia/pylaia-himanis**                            | Hugging Face + PyLaia                                  |        **Sangat cocok untuk medieval Latin/French-like material** | Dilatih pada handwritten medieval French dan Latin; line-level; aspect ratio dipertahankan; tersedia character LM | Hanya berguna bila skrip dan convention cukup dekat; output charset mungkin tidak sama                                                           | **Rendah–sedang** | **Sedang**        | **P1 jika domain dekat**             |
| **Teklia/pylaia-belfort**                            | Hugging Face + PyLaia                                  |                **Cocok untuk historical French-like handwriting** | Historical handwritten checkpoint; line recognizer siap fine-tune                                                 | Domain bahasa dan tulisan spesifik; tidak otomatis transfer ke semua manuskrip                                                                   | **Rendah–sedang** | **Sedang**        | **P1–P2**                            |
| **Teklia/pylaia-norhand-v1/v3**                      | Hugging Face + PyLaia                                  |                          **Cocok untuk Latin-script handwriting** | Banyak handwritten lines; preserve aspect ratio; tersedia baseline CER/WER dan LM                                 | Norwegian language prior; bentuk tulisan modern mungkin berbeda dari historical material                                                         | **Rendah–sedang** | **Sedang**        | **P2**, atau P1 jika visualnya dekat |
| **Teklia/pylaia-iam**                                | Hugging Face + PyLaia                                  |        **Cocok sebagai generic Latin handwriting initialization** | Dataset IAM line-level; cocok sebagai pembanding dengan TrOCR IAM                                                 | Modern English handwriting, bukan historical; visual domain gap mungkin besar                                                                    | **Rendah–sedang** | **Sedang**        | **P2**                               |
| **Kraken CATMuS Medieval**                           | Kraken model repository, DOI `10.5281/zenodo.12743230` |      **Sangat cocok untuk medieval Western European manuscripts** | Dilatih pada Old/Middle French, Latin, Spanish, dan Italian; dibuat sebagai generic historical HTR model          | Transcription guideline menggunakan graphematic transcription dan NFD; charset/normalisasi harus disejajarkan                                    | **Rendah–sedang** | **Sedang–tinggi** | **P1 jika periode cocok**            |
| **Kraken McCATMuS**                                  | Kraken model repository                                |                  **Cocok untuk dokumen abad ke-16 hingga modern** | Generic recognition model untuk handwritten, printed, dan typewritten documents; coverage luas                    | Generality dapat menurunkan specialization; perlu fine-tuning target domain                                                                      | **Rendah–sedang** | **Sedang–tinggi** | **P1–P2**                            |
| **Kraken TRIDIS**                                    | Zenodo/Kraken model                                    | **Cocok untuk medieval dan Early Modern documentary manuscripts** | Domain historical sangat dekat untuk legal/administrative manuscripts                                             | Transcription convention dan bahasa harus cocok; workflow bukan HF-native                                                                        | **Rendah–sedang** | **Sedang–tinggi** | **P1 jika domain dekat**             |
| **PaddleOCR/SVTR/PP-OCR recognizer**                 | PaddleOCR                                              |                                   **Kurang cocok out-of-the-box** | Ekosistem matang dan inference cepat                                                                              | Mayoritas checkpoint berfokus printed/scene text; width dan maximum text length sering dikonfigurasi untuk word crops, bukan 120-character lines | **Rendah–sedang** | **Sedang**        | **P3**                               |
| **PARSeq/ABINet/SATRN pretrained scene-text models** | Official implementations, MMOCR/docTR                  |                     **Kurang cocok sebagai starting point utama** | Visual-language modeling kuat; tersedia pretrained weights                                                        | Sering dilatih pada short word crops; maximum label length dan image aspect ratio tidak sesuai long handwritten lines                            | **Sedang**        | **Sedang–tinggi** | **P3**                               |

`Teklia/pylaia-himanis` secara khusus dilatih pada handwritten medieval French dan Latin. Model tersebut menggunakan fixed image height sambil mempertahankan original aspect ratio. ([Hugging Face][7])

CATMuS Medieval adalah Kraken HTR model untuk Old/Middle French, Latin, Spanish, dan Italian. Model ini berpotensi sangat kuat, tetapi menggunakan graphematic transcription dan normalisasi Unicode NFD, sehingga aturan label kompetisi harus diperiksa sebelum transfer learning. ([Zenodo][8])

---

## Ringkasan prioritas keseluruhan

| Urutan | Eksperimen                                              | Alasan                                                                               |
| -----: | ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
|  **1** | Custom CRNN-BiLSTM-CTC                                  | Baseline paling stabil, murah, dan cocok dengan line images sangat lebar             |
|  **2** | TrOCR-Small-Handwritten default                         | Baseline pretrained Transformer tercepat untuk dibuat                                |
|  **3** | TrOCR-Small aspect-ratio-aware                          | Kemungkinan memberikan perbaikan besar dibanding resize persegi                      |
|  **4** | CRNN encoder lebih kuat, misalnya ResNet + BiLSTM + CTC | Meningkatkan visual representation tanpa compute terlalu besar                       |
|  **5** | PyLaia/Kraken checkpoint yang paling dekat domain       | Sangat potensial bila script, bahasa, era, dan transcription convention cocok        |
|  **6** | TrOCR-Base dengan partial freezing                      | Kandidat accuracy tinggi dengan risiko compute dan overfitting yang masih terkontrol |
|  **7** | CTC + character LM / lexicon-free beam search           | Terutama untuk memperbaiki WER                                                       |
|  **8** | Ensemble CRNN + TrOCR                                   | CTC dan autoregressive decoder cenderung menghasilkan error yang berbeda             |
|  **9** | TrOCR-Large                                             | Hanya bila compute cukup dan Base jelas menunjukkan scaling benefit                  |
| **10** | Donut-Base                                              | Eksperimen tambahan dengan peluang keuntungan lebih rendah                           |

---

## Rekomendasi final per pendekatan

| Pendekatan                 | Model yang dipilih                                          | Keputusan                              |
| -------------------------- | ----------------------------------------------------------- | -------------------------------------- |
| **CRNN + CTC**             | ResNet-small + 2-layer BiLSTM + character CTC               | **Wajib dijalankan**                   |
| **TrOCR**                  | `microsoft/trocr-small-handwritten`, kemudian Base          | **Jalur pretrained utama**             |
| **Donut**                  | `naver-clova-ix/donut-base`, bukan CORD/DocVQA              | **Prioritas rendah**                   |
| **Pretrained HTR library** | PyLaia Himanis atau Kraken CATMuS/TRIDIS, tergantung domain | **Prioritas tinggi bila domain cocok** |

Secara praktis, kandidat final paling kuat kemungkinan bukan satu model tunggal, melainkan **ensemble CRNN-CTC yang kuat secara visual dengan TrOCR yang kuat secara language modeling**. CRNN biasanya lebih literal terhadap karakter yang terlihat, sedangkan TrOCR lebih baik dalam menghasilkan urutan kata yang masuk akal, tetapi lebih berisiko melakukan substitusi atau hallucination.

[1]: https://huggingface.co/microsoft/trocr-base-handwritten/raw/main/preprocessor_config.json "huggingface.co"
[2]: https://doc.teklia.com/pylaia/ "PyLaia :: Teklia Documentation"
[3]: https://kraken.re/5.3.0/index.html "kraken — kraken  documentation"
[4]: https://huggingface.co/Teklia/pylaia-norhand-v1?utm_source=chatgpt.com "Teklia/pylaia-norhand-v1"
[5]: https://huggingface.co/microsoft/trocr-small-handwritten "microsoft/trocr-small-handwritten · Hugging Face"
[6]: https://github.com/clovaai/donut "GitHub - clovaai/donut: Official Implementation of OCR-free Document Understanding Transformer (Donut) and Synthetic Document Generator (SynthDoG), ECCV 2022 · GitHub"
[7]: https://huggingface.co/Teklia/pylaia-himanis?utm_source=chatgpt.com "Teklia/pylaia-himanis"
[8]: https://zenodo.org/records/12743230?utm_source=chatgpt.com "CATMuS Medieval"

---

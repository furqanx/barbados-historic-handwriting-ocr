Untuk CTC decoding, pilihannya bertingkat seperti ini:

**2. Beam Search Tanpa LM**
Model menyimpan beberapa kandidat teks terbaik, bukan cuma satu jalur terbaik.

Contoh:

```text
beam_size = 5 / 10 / 25
```

Kelebihan:
- masih tidak butuh training ulang,
- bisa memperbaiki deletion/insertion,
- relatif aman untuk simbol langka karena belum ada language bias besar.

Ini opsi pertama yang saya sarankan.

**3. Beam Search + Character LM**
Beam search diberi bantuan language model level karakter.

LM belajar pola teks dari `Train.csv`, misalnya:
- kombinasi huruf umum,
- spasi,
- akhiran kata,
- tanda baca,
- singkatan historis.

Kelebihan:
- bisa memperbaiki kata dan spacing,
- hemat GPU,
- cocok untuk dataset kecil.

Risiko:
- LM terlalu kuat bisa menghapus simbol langka seperti `^`, `*`, `#`.
- Jadi bobot LM harus ringan.

**5. Reranking Kandidat**
Generate beberapa kandidat dari beam search, lalu pilih ulang memakai aturan/score tambahan.

Contoh reranker:
- penalti output terlalu pendek,
- reward simbol jika visual model memberi probabilitas tinggi,
- penalti spacing aneh,
- pilih kandidat yang panjangnya paling masuk akal.

Ini lebih advanced, tapi potensial.


Library yang bisa dipakai:
- `pyctcdecode`
- `flashlight-text`
- implementasi beam search custom ringan

Untuk kasus kita, saya condong mulai dari **custom prefix beam search ringan** atau `pyctcdecode` tanpa lexicon dulu.

Kandidat yang layak kita implementasikan:

1. **Beam Search Tanpa LM**
   - Prioritas pertama.
   - Risiko rendah, GPU hampir nol.

2. **Beam Search + Character LM Ringan**
   - Prioritas kedua.
   - Potensi naik lebih besar, tapi perlu kontrol agar simbol langka tidak hilang.

3. **Reranking Kandidat**
   - Prioritas ketiga.
   - Dipakai setelah beam search menghasilkan beberapa kandidat.

Yang kita skip dulu:

- **Greedy Decode**: sudah ada.
- **Word Lexicon**: skip dulu, terlalu berisiko untuk teks historis dan nama unik.

---

Prioritas praktis:

1. **CTC beam search tanpa LM**
2. **CTC beam search + char LM ringan**
3. **CTC reranking**
4. **PyLaia/Kraken historical HTR satu checkpoint paling dekat**
5. **TrOCR hanya untuk ensemble/diversity**, bukan tuning besar-besaran dulu

Ini lebih hemat credit dan lebih selaras dengan sinyal public score.









---

Maksudnya: sekarang semua model terbaik kita baru dilatih pada **fold 0**.

Dalam cross-validation 5 fold, data train dibagi seperti ini:

```text
fold 0
fold 1
fold 2
fold 3
fold 4
```

Saat training fold 0:

```text
validasi = fold 0
training = fold 1 + 2 + 3 + 4
```

Kalau kita train fold 1:

```text
validasi = fold 1
training = fold 0 + 2 + 3 + 4
```

Model fold 0 dan fold 1 akan belajar dari komposisi data yang berbeda. Saat inference test, prediksi mereka bisa sedikit berbeda. Lalu kita ensemble:

```text
prediction_fold0 + prediction_fold1 -> submission ensemble
```

Kenapa ini bisa membantu?

Karena multi-fold biasanya membuat prediksi lebih stabil:
- mengurangi bias dari satu split;
- model melihat variasi train/valid berbeda;
- error antar fold bisa saling menutup.

Kenapa jangan langsung 5 fold?

Karena mahal GPU. Kalau satu ResNet butuh misalnya 2 jam:

```text
5 fold = sekitar 10 jam
```

Belum tentu peningkatannya besar.

Jadi strategi hemat:

```text
1. Kita sudah punya ResNet fold 0.
2. Train ResNet fold 1 dengan konfigurasi terbaik.
3. Ensemble fold 0 + fold 1.
4. Submit.
5. Kalau naik cukup jelas, baru lanjut fold 2, 3, 4.
```

Dengan kata lain: **uji dulu apakah multi-fold memberi sinyal positif sebelum membakar banyak compute**.








---

Maksudnya: setelah model menghasilkan teks, kita bisa melakukan **perbaikan kecil pada output prediksi** sebelum dibuat submission.

Contoh model memprediksi:

```text
"Sealed  and delivered"
```

Kita ubah jadi:

```text
"Sealed and delivered"
```

Atau model sering menghasilkan:

```text
"St .Phillips"
```

Kita koreksi jadi:

```text
"St.Phillips"
```

Yang mungkin bisa dikoreksi:

- spasi ganda
- spasi sebelum tanda baca
- spasi setelah simbol tertentu
- punctuation yang sering salah
- singkatan historis seperti `W^m`, `St.`, `Jn.`
- output terlalu pendek karena deletion umum
- karakter yang sering tertukar, misalnya `1/l`, `0/O`, jika terbukti dari validation

Tapi bagian pentingnya adalah:

```text
harus divalidasi, jangan rule asal
```

Artinya setiap rule harus diuji dulu pada validation prediction yang punya ground truth.

Contoh:

```text
Rule: hapus spasi sebelum titik
```

Kita cek di validation:

```text
sebelum rule: CER 0.09062
sesudah rule: CER 0.08990
```

Kalau membaik, pakai. Kalau memburuk, buang.

Jangan membuat rule hanya karena “terlihat masuk akal”, karena teks historis sering punya format aneh yang justru benar.







---

Maksudnya: kita **tidak melatih model baru**, tapi memakai hasil prediksi dari model-model yang sudah ada, lalu menggabungkannya menjadi satu submission baru.

Contoh kita punya 4 file prediksi:

```text
resnet_h96_w2048_submission.csv
resnet_h96_w3072_submission.csv
crnn_best_submission.csv
trocr_large_submission.csv
```

Setiap file punya prediksi untuk ID yang sama:

```text
ID        ResNet2048        ResNet3072        CRNN           TrOCR
abc       "John Smith"      "John Smith"      "John Smlth"   "John Smith"
xyz       "W^m Booke"       "Wm Booke"        "W^m Booke"    "Wm Book"
```

Ensemble memilih hasil akhir, misalnya:
- majority vote: pilih teks yang paling sering muncul;
- priority vote: kalau berbeda, lebih percaya ResNet;
- rule-based: untuk simbol langka lebih percaya CTC;
- validation-based: pilih strategi yang terbukti paling bagus di validation.

Contoh hasil ensemble:

```text
abc -> "John Smith"   karena 3 model setuju
xyz -> "W^m Booke"    karena CTC lebih kuat untuk simbol langka
```

Kenapa murah?

Karena kita hanya membaca CSV dan memilih/menggabungkan teks. Tidak butuh GPU, tidak butuh training.

Kenapa mulai dari validation-based ensemble?

Karena kita punya validation prediction dengan ground truth. Jadi kita bisa uji beberapa strategi ensemble di validation dulu:

```text
strategi A: majority vote
strategi B: prioritas ResNet
strategi C: prioritas TrOCR kalau line pendek
strategi D: prioritas CTC kalau ada simbol
```

Lalu lihat strategi mana yang benar-benar menurunkan CER/WER lokal. Setelah itu baru diterapkan ke file test/submission.
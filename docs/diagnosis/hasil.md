Maksudnya begini:

Untuk CTC, model butuh cukup banyak **time step horizontal** agar bisa menulis seluruh karakter target. Hasil audit menunjukkan:

```text
Invalid CTC rows: 0
```

Artinya dengan setting:

```text
target-height = 96
max-width = 2048
stride = 4
```

tidak ada sampel yang secara matematis mustahil dipelajari oleh CTC. Ini kabar bagus.

`Minimum CTC margin: 26` berarti sampel paling “mepet” masih punya 26 time step ekstra setelah kebutuhan minimal terpenuhi. Jadi dari sisi alignment, aman.

Tapi:

```text
Width clipped rows: 979
```

berarti ada 979 gambar yang setelah resize seharusnya lebih lebar dari 2048, lalu dipotong/dikecilkan ke batas `max-width=2048`.

Dampaknya:
- teks masih cukup panjang untuk CTC,
- tapi detail visual bisa lebih padat,
- simbol kecil/tanda baca bisa lebih sulit terbaca,
- line panjang mungkin lebih rawan error.

Kesimpulan untuk training selanjutnya:

1. **CRNN/ResNet/ConvNeXt CTC dengan `h96/w2048` boleh dilanjutkan.**
   Pipeline-nya valid, tidak ada masalah alignment fatal.

2. **Jangan ubah stride dulu.**
   `stride=4` sudah aman.

3. **Eksperimen yang masuk akal berikutnya adalah `max-width` lebih besar**, misalnya:
   ```text
   h96 / w3072
   ```
   atau
   ```text
   h128 / w3072
   ```

4. Kalau GPU memory terbatas, tetap mulai dari:
   ```text
   target-height=96
   max-width=2048
   ```
   sebagai baseline sehat.

5. Kalau ingin mengejar simbol kecil dan line panjang, coba:
   ```text
   target-height=128
   max-width=3072
   batch-size lebih kecil
   ```

Jadi keputusan praktisnya: **CTC baseline aman untuk training, tapi ablation resolusi/width masih penting.**

---

Maksudnya begini:

Untuk TrOCR ada **dua hal berbeda** yang diaudit:

**1. Panjang teks target**
Ini aman.

```text
max token count = 35
max limit = 192
```

Artinya label/transkripsi tidak akan kepotong saat training atau generation. Jadi `max_label_length=192` dan `max_generation_length=192` sudah sangat cukup.

**2. Ukuran gambar aspect-aware**
Ini bermasalah.

Kita pakai:

```text
target-height = 384
canvas-width = 1536
```

Saat gambar dinaikkan ke tinggi 384 sambil mempertahankan aspect ratio, banyak gambar menjadi jauh lebih lebar dari 1536. Karena canvas cuma 1536, gambar akan dipotong secara horizontal.

Contoh sederhana:

```text
original aspect ratio = 17
target-height = 384
resized width ≈ 17 × 384 = 6528 px
canvas-width = 1536 px
```

Jadi bagian kanan line bisa hilang. Kalau teks di kanan hilang, model mustahil belajar transkripsi penuh.

Kesimpulan untuk training selanjutnya:

- **Jangan pakai TrOCR aspect-aware dengan `target-height=384` dan `canvas-width=1536`**.
- Untuk TrOCR, ada dua opsi yang lebih aman:

1. Pakai **TrOCR default preprocessing** dulu.
   Ini tidak ideal untuk line panjang, tapi tidak memotong canvas custom kita.

2. Kalau tetap mau aspect-aware, turunkan tinggi atau naikkan canvas:
   - `target-height=96` atau `128`
   - `canvas-width=2048`, `3072`, atau lebih
   - lalu jalankan ulang `diagnose_trocr_targets.py`

Rekomendasi praktis saya:

```text
TrOCR next experiment:
target-height = 128
canvas-width = 2048 atau 3072
```

Lalu audit lagi. Targetnya:

```text
aspect_canvas_clipped_rows mendekati 0
```

Kalau masih banyak clipping, jangan training dulu.
Menurut saya **perlu 2-3 training dengan hyperparameter berbeda**, karena baseline CRNN-CTC dari random init cukup sensitif terhadap ukuran gambar, batch size, dan learning rate.

Dari log Anda, epoch awal masih sangat buruk (`WER/CER ~ 1.0`), itu normal untuk CTC. Jangan langsung simpulkan gagal sebelum beberapa epoch.

Coba 3 run ini:

**Run 1: baseline sekarang**
```bash
--run-name crnn_ctc_h96_w2048_bs16_lr1e3_fold0 \
--epochs 30 \
--batch-size 16 \
--target-height 96 \
--max-width 2048 \
--lr 1e-3
```

**Run 2: image lebih kecil, lebih cepat**
```bash
--run-name crnn_ctc_h64_w1536_bs24_lr1e3_fold0 \
--epochs 30 \
--batch-size 24 \
--target-height 64 \
--max-width 1536 \
--lr 1e-3
```

**Run 3: learning rate lebih konservatif**
```bash
--run-name crnn_ctc_h96_w2048_bs16_lr3e4_fold0 \
--epochs 30 \
--batch-size 16 \
--target-height 96 \
--max-width 2048 \
--lr 3e-4
```

Prioritas saya:
1. lanjutkan run yang sedang berjalan sampai minimal `20-30 epoch`
2. jalankan Run 2 untuk pembanding cepat
3. jalankan Run 3 kalau Run 1 loss tidak stabil atau prediksi terlalu kosong/berulang

Untuk CTC, biasanya perbaikan baru terlihat setelah loss turun cukup jauh, jadi jangan terlalu percaya epoch 1-3.
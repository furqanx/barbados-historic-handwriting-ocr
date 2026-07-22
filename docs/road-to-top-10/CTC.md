Iya, untuk **fase CTC decoding** sudah settled.

Eksperimen utama yang akan dijalankan ada **6**:

1. `beam_size=10`, tanpa LM
2. `beam_size=25`, tanpa LM
3. `beam_size=25`, `lm_weight=0.02`
4. `beam_size=25`, `lm_weight=0.05`
5. `beam_size=25`, `lm_weight=0.10`
6. konfigurasi terbaik + rerank ringan

Tambahan pendukung:

- train character LM `order=4`
- evaluasi validation untuk setiap hasil

Soal GPU:

- **Train character LM**: tidak perlu GPU, CPU saja.
- **Beam/beam+LM/rerank decoding**: tidak wajib GPU, tapi **lebih baik pakai GPU** karena tetap perlu forward pass model ResNet untuk menghasilkan logits.
- **Evaluasi validation**: CPU saja.
- **Reranking setelah kandidat ada**: CPU saja.

Jadi yang memakai GPU hanya tahap `predict_ctc.py` karena dia menjalankan checkpoint ResNet ke gambar. Bagian LM, evaluator, dan reranking logic-nya ringan.
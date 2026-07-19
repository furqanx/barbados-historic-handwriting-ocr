Berikut seluruh alur pelatihan model yang sekarang ada di workspace kita.

**1. CRNN + BiLSTM + CTC**
- Script train: `scripts/train_crnn_ctc.py`
- Script predict: `scripts/predict_crnn_ctc.py`
- Arsitektur custom PyTorch dari nol.
- Cocok sebagai baseline cepat dan ringan.
- Hyperparameter utama: `target-height`, `max-width`, `batch-size`, `lr`, `epochs`, `fold`.

**2. ResNet-style CNN + BiLSTM + CTC**
- Script train: `scripts/train_resnet_ctc.py`
- Script predict: `scripts/predict_ctc.py`
- Encoder lebih kuat dari CRNN awal, masih custom PyTorch.
- Cocok untuk baseline CTC yang lebih serius.
- Hyperparameter utama: `base-channels`, `target-height`, `max-width`, `batch-size`, `lr`.

**3. ConvNeXt-tiny + BiLSTM + CTC**
- Script train: `scripts/train_convnext_ctc.py`
- Script predict: `scripts/predict_ctc.py`
- Menggunakan `timm` ConvNeXt-tiny encoder, bisa pretrained ImageNet.
- Jalur CTC paling modern di kode kita.
- Hyperparameter utama: `backbone-name`, `target-height`, `max-width`, `batch-size`, `lr`, `rnn-hidden-size`.

**4. TrOCR-Small default**
- Script train: `scripts/train_trocr.py`
- Script predict: `scripts/predict_trocr.py`
- Model: `microsoft/trocr-small-handwritten`
- Pretrained handwriting IAM.
- Cocok sebagai autoregressive baseline paling realistis.

**5. TrOCR-Small aspect-ratio-aware**
- Script train: `scripts/train_trocr.py`
- Script predict: `scripts/predict_trocr.py`
- Model sama, tapi preprocessing preserve aspect ratio dengan canvas width.
- Hyperparameter penting: `--preprocess-mode aspect`, `--target-height`, `--canvas-width`.
- Ini varian TrOCR yang paling sesuai dengan line image panjang.

**6. TrOCR-Base dengan partial freezing**
- Script train: `scripts/train_trocr.py`
- Script predict: `scripts/predict_trocr.py`
- Model: `microsoft/trocr-base-handwritten`
- Bisa freeze sebagian encoder/decoder.
- Hyperparameter penting: `--freeze-encoder-layers`, `batch-size`, `lr`.

**7. TrOCR-Large**
- Script train: `scripts/train_trocr.py`
- Script predict: `scripts/predict_trocr.py`
- Model: `microsoft/trocr-large-handwritten`
- Paling berat, sebaiknya hanya kalau GPU cukup.
- Hyperparameter utama: `batch-size 1`, `lr kecil`, `epochs sedikit`.

**8. Ensemble CSV**
- Script: `scripts/ensemble_predictions.py`
- Menggabungkan hasil prediksi beberapa model.
- Bisa ensemble CRNN/ResNet/ConvNeXt/TrOCR/PyLaia/Kraken/PaddleOCR selama output-nya CSV `ID,Target`.
- Cocok dilakukan di akhir.

**9. PyLaia HTR Workflow**
- Prepare dataset: `scripts/prepare_pylaia_dataset.py`
- Download model: `scripts/download_pylaia_model.py`
- Audit charset: `scripts/audit_pylaia_symbols.py`
- Validate dataset: `scripts/validate_pylaia_dataset.py`
- Train: `scripts/train_pylaia.py`
- Predict: `scripts/predict_pylaia.py`
- Convert submission: `scripts/convert_pylaia_predictions.py`
- Model kandidat: `Teklia/pylaia-himanis`, `belfort`, `norhand`, `iam`.
- Cocok untuk historical handwriting line recognition.

**10. Kraken Historical HTR Workflow**
- Prepare dataset: `scripts/prepare_kraken_dataset.py`
- Download model: `scripts/download_kraken_model.py`
- Audit text: `scripts/audit_kraken_text.py`
- Train: `scripts/train_kraken.py`
- Test/evaluate: `scripts/test_kraken.py`
- Predict: `scripts/predict_kraken.py`
- Convert submission: `scripts/convert_kraken_predictions.py`
- Model kandidat: CATMuS Medieval, McCATMuS, TRIDIS.
- Cocok untuk historical manuscripts, tapi dependency/runtime lebih sensitif.

**11. PaddleOCR / SVTR / PP-OCR Recognizer**
- Prepare dataset: `scripts/prepare_paddleocr_dataset.py`
- Audit text: `scripts/audit_paddleocr_text.py`
- Patch config: `scripts/make_paddleocr_config.py`
- Train: `scripts/train_paddleocr_rec.py`
- Eval: `scripts/eval_paddleocr_rec.py`
- Predict: `scripts/predict_paddleocr_rec.py`
- Convert submission: `scripts/convert_paddleocr_predictions.py`
- Ini workflow P3/cadangan, karena PaddleOCR lebih umum untuk OCR pendek/printed/scene text.
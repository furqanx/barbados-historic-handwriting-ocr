from src.paddleocr_rec.config import patch_recognition_config


def test_patch_recognition_config_updates_common_keys() -> None:
    config = {
        "Global": {},
        "Train": {
            "dataset": {
                "transforms": [{"RecResizeImg": {"image_shape": [3, 48, 320]}}],
            },
            "loader": {},
        },
        "Eval": {
            "dataset": {
                "transforms": [{"RecResizeImg": {"image_shape": [3, 48, 320]}}],
            },
            "loader": {},
        },
    }

    patched = patch_recognition_config(
        config,
        train_labels="train.txt",
        val_labels="val.txt",
        character_dict="dict.txt",
        save_model_dir="outputs",
        max_text_length=128,
        image_shape="3,64,2048",
        train_batch_size=8,
        eval_batch_size=4,
        epoch_num=10,
        use_amp=True,
    )

    assert patched["Global"]["character_dict_path"] == "dict.txt"
    assert patched["Global"]["use_space_char"] is True
    assert patched["Global"]["epoch_num"] == 10
    assert patched["Train"]["dataset"]["label_file_list"] == ["train.txt"]
    assert patched["Eval"]["dataset"]["label_file_list"] == ["val.txt"]
    assert patched["Train"]["loader"]["batch_size_per_card"] == 8
    assert patched["Eval"]["loader"]["batch_size_per_card"] == 4
    assert patched["Train"]["dataset"]["transforms"][0]["RecResizeImg"]["image_shape"] == [
        3,
        64,
        2048,
    ]


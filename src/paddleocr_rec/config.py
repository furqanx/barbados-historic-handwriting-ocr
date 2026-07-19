"""Patch PaddleOCR YAML configs for local recognition experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with PyYAML."""

    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping YAML config: {path}")
    return payload


def save_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    """Save a YAML mapping."""

    import yaml

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output


def patch_recognition_config(
    config: dict[str, Any],
    *,
    train_labels: str | Path,
    val_labels: str | Path,
    character_dict: str | Path,
    save_model_dir: str | Path,
    pretrained_model: str | Path | None = None,
    max_text_length: int = 128,
    image_shape: str | None = "3,64,2048",
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    epoch_num: int | None = None,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """Patch common PaddleOCR recognition config keys."""

    patched = _deepcopy(config)
    global_cfg = _ensure_mapping(patched, "Global")
    global_cfg["character_dict_path"] = str(character_dict)
    global_cfg["use_space_char"] = True
    global_cfg["max_text_length"] = int(max_text_length)
    global_cfg["save_model_dir"] = str(save_model_dir)
    if pretrained_model is not None:
        global_cfg["pretrained_model"] = str(pretrained_model)
    if epoch_num is not None:
        global_cfg["epoch_num"] = int(epoch_num)
    if use_amp is not None:
        global_cfg["use_amp"] = bool(use_amp)

    train_cfg = _ensure_mapping(patched, "Train")
    eval_cfg = _ensure_mapping(patched, "Eval")
    _patch_dataset(train_cfg, train_labels)
    _patch_dataset(eval_cfg, val_labels)
    if train_batch_size is not None:
        _ensure_mapping(train_cfg, "loader")["batch_size_per_card"] = int(train_batch_size)
    if eval_batch_size is not None:
        _ensure_mapping(eval_cfg, "loader")["batch_size_per_card"] = int(eval_batch_size)
    if image_shape is not None:
        _patch_rec_resize_image_shape(patched, _parse_image_shape(image_shape))
    return patched


def _patch_dataset(split_cfg: dict[str, Any], label_path: str | Path) -> None:
    dataset = _ensure_mapping(split_cfg, "dataset")
    dataset["name"] = dataset.get("name", "SimpleDataSet")
    dataset["label_file_list"] = [str(label_path)]


def _patch_rec_resize_image_shape(config: dict[str, Any], image_shape: list[int]) -> None:
    for split_name in ("Train", "Eval"):
        split_cfg = config.get(split_name)
        if not isinstance(split_cfg, dict):
            continue
        dataset = split_cfg.get("dataset")
        if not isinstance(dataset, dict):
            continue
        transforms = dataset.get("transforms")
        if not isinstance(transforms, list):
            continue
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            rec_resize = transform.get("RecResizeImg")
            if isinstance(rec_resize, dict):
                rec_resize["image_shape"] = image_shape


def _parse_image_shape(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",")]


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config field must be a mapping: {key}")
    return value


def _deepcopy(payload: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(payload)


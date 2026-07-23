"""Orchestrate Kraken HTR screening, fine-tuning, and recognition grids."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (  # noqa: E402
    KRAKEN_DATA_DIR,
    KRAKEN_MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLE_SUBMISSION_CSV,
    TRAIN_MANIFEST,
)
from src.htr.kraken.model_registry import KRAKEN_MODEL_REGISTRY, resolve_kraken_model  # noqa: E402


@dataclass(frozen=True)
class DecodeProfile:
    """A named Kraken recognition profile."""

    name: str
    extra_args: tuple[str, ...] = ()


DEFAULT_DECODE_PROFILES: dict[str, DecodeProfile] = {
    "native": DecodeProfile(name="native"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(KRAKEN_MODEL_REGISTRY),
        choices=sorted(KRAKEN_MODEL_REGISTRY),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--phase",
        choices=["zeroshot-small", "zeroshot", "train", "decode", "test", "all"],
        default="zeroshot-small",
    )
    parser.add_argument("--decode-profile", action="append", default=None)
    parser.add_argument(
        "--profile-extra-arg",
        action="append",
        default=[],
        metavar="PROFILE:ARG",
        help="Attach raw Kraken OCR args to a profile. Repeat per token.",
    )
    parser.add_argument(
        "--dataset-template",
        default=str(KRAKEN_DATA_DIR / "fold{fold}_h128_{unicode_normalization}"),
        help="Dataset directory template. Keys: {fold}, {unicode_normalization}.",
    )
    parser.add_argument(
        "--model-path-template",
        default=str(KRAKEN_MODELS_DIR / "{model}" / "{filename}"),
        help="Model path template. Keys: {model}, {filename}.",
    )
    parser.add_argument(
        "--run-template",
        default="kraken_{model}_fold{fold}_h128_{unicode_normalization}",
        help="Fine-tuned run name template. Keys: {model}, {fold}, {unicode_normalization}.",
    )
    parser.add_argument(
        "--zeroshot-run-template",
        default="kraken_{model}_zeroshot_fold{fold}_h128_{unicode_normalization}",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--reference-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _decode_profiles(args)
    for fold in args.folds:
        for model_key in args.models:
            model_info = resolve_kraken_model(model_key)
            unicode_normalization = model_info.unicode_normalization
            dataset_dir = _format_path(
                args.dataset_template,
                fold=fold,
                unicode_normalization=unicode_normalization.lower(),
            )
            model_path = _format_path(
                args.model_path_template,
                model=model_key,
                filename=model_info.filename,
            )
            _require_files([dataset_dir, model_path])
            run_values = {
                "model": model_key,
                "fold": fold,
                "unicode_normalization": unicode_normalization.lower(),
            }
            if args.phase in {"zeroshot-small", "zeroshot"}:
                base_run = args.zeroshot_run_template.format(**run_values)
                max_images = args.max_images if args.phase == "zeroshot-small" else None
                _run_decode_grid(
                    args,
                    profiles,
                    base_run,
                    dataset_dir,
                    model_path,
                    unicode_normalization=unicode_normalization,
                    split="val",
                    max_images=max_images,
                )
            elif args.phase in {"train", "all"}:
                run_name = args.run_template.format(**run_values)
                _run(
                    _train_command(
                        args,
                        run_name,
                        dataset_dir,
                        model_path,
                        unicode_normalization=unicode_normalization,
                    ),
                    args.dry_run,
                )
                if args.phase == "all":
                    fine_tuned_model = OUTPUTS_DIR / "kraken" / run_name / f"{run_name}_best.mlmodel"
                    model_for_decode = fine_tuned_model if fine_tuned_model.exists() else model_path
                    _run_decode_grid(
                        args,
                        profiles,
                        run_name,
                        dataset_dir,
                        model_for_decode,
                        unicode_normalization=unicode_normalization,
                        split="val",
                        max_images=None,
                    )
                    _run_decode_grid(
                        args,
                        profiles,
                        run_name,
                        dataset_dir,
                        model_for_decode,
                        unicode_normalization=unicode_normalization,
                        split="test",
                        max_images=None,
                    )
            elif args.phase == "decode":
                run_name = args.run_template.format(**run_values)
                fine_tuned_model = OUTPUTS_DIR / "kraken" / run_name / f"{run_name}_best.mlmodel"
                _require_files([fine_tuned_model])
                _run_decode_grid(
                    args,
                    profiles,
                    run_name,
                    dataset_dir,
                    fine_tuned_model,
                    unicode_normalization=unicode_normalization,
                    split="val",
                    max_images=None,
                )
            elif args.phase == "test":
                run_name = args.run_template.format(**run_values)
                fine_tuned_model = OUTPUTS_DIR / "kraken" / run_name / f"{run_name}_best.mlmodel"
                _require_files([fine_tuned_model])
                _run_decode_grid(
                    args,
                    profiles,
                    run_name,
                    dataset_dir,
                    fine_tuned_model,
                    unicode_normalization=unicode_normalization,
                    split="test",
                    max_images=None,
                )


def _run_decode_grid(
    args: argparse.Namespace,
    profiles: list[DecodeProfile],
    base_run_name: str,
    dataset_dir: Path,
    model_path: Path,
    *,
    unicode_normalization: str,
    split: str,
    max_images: int | None,
) -> None:
    for profile in profiles:
        run_name = f"{base_run_name}_{profile.name}"
        raw_dir = args.output_dir / "kraken" / run_name / f"{split}_raw"
        _run(
            _predict_command(
                args,
                run_name=run_name,
                dataset_dir=dataset_dir,
                model_path=model_path,
                split=split,
                raw_dir=raw_dir,
                profile=profile,
                max_images=max_images,
            ),
            args.dry_run,
        )
        _run(
            _convert_command(
                args,
                run_name=run_name,
                split=split,
                raw_dir=raw_dir,
                unicode_normalization=unicode_normalization,
            ),
            args.dry_run,
        )


def _train_command(
    args: argparse.Namespace,
    run_name: str,
    dataset_dir: Path,
    model_path: Path,
    *,
    unicode_normalization: str,
) -> list[str]:
    return [
        args.python,
        str(PROJECT_ROOT / "scripts" / "train_kraken.py"),
        "--run-name",
        run_name,
        "--dataset-dir",
        str(dataset_dir),
        "--base-model",
        str(model_path),
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--lag",
        str(args.lag),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--unicode-normalization",
        unicode_normalization,
        "--device",
        args.device,
        "--precision",
        args.precision,
    ]


def _predict_command(
    args: argparse.Namespace,
    *,
    run_name: str,
    dataset_dir: Path,
    model_path: Path,
    split: str,
    raw_dir: Path,
    profile: DecodeProfile,
    max_images: int | None,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "predict_kraken.py"),
        "--run-name",
        run_name,
        "--dataset-dir",
        str(dataset_dir),
        "--model",
        str(model_path),
        "--split",
        split,
        "--output-dir",
        str(raw_dir),
        "--start-index",
        str(args.start_index),
    ]
    if max_images is not None:
        command.extend(["--max-images", str(max_images)])
    for extra_arg in profile.extra_args:
        command.extend(["--extra-arg", extra_arg])
    return command


def _convert_command(
    args: argparse.Namespace,
    *,
    run_name: str,
    split: str,
    raw_dir: Path,
    unicode_normalization: str,
) -> list[str]:
    output_normalization = "NFC" if unicode_normalization == "NFD" else unicode_normalization
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "convert_kraken_predictions.py"),
        "--run-name",
        run_name,
        "--split",
        split,
        "--prediction-dir",
        str(raw_dir),
        "--output-dir",
        str(args.output_dir),
        "--output-unicode-normalization",
        output_normalization,
    ]
    if split == "test":
        command.extend(["--sample-submission", str(args.sample_submission)])
    else:
        command.extend(["--reference-manifest", str(args.reference_manifest)])
    return command


def _decode_profiles(args: argparse.Namespace) -> list[DecodeProfile]:
    requested = args.decode_profile or ["native"]
    profile_args: dict[str, list[str]] = {name: [] for name in requested}
    for item in args.profile_extra_arg:
        if ":" not in item:
            raise ValueError(f"Invalid --profile-extra-arg {item!r}; use PROFILE:ARG.")
        name, extra_arg = item.split(":", 1)
        profile_args.setdefault(name, []).append(extra_arg)
    profiles = []
    for name in requested:
        if name not in DEFAULT_DECODE_PROFILES and not profile_args.get(name):
            raise ValueError(
                "Kraken does not expose a portable beam/LM profile in the documented "
                "recognition CLI. Use --decode-profile native, or provide explicit "
                "runtime-specific args with --profile-extra-arg PROFILE:ARG."
            )
        default = DEFAULT_DECODE_PROFILES.get(name, DecodeProfile(name=name))
        profiles.append(DecodeProfile(name=name, extra_args=tuple(default.extra_args + tuple(profile_args.get(name, [])))))
    return profiles


def _format_path(template: str, **values: object) -> Path:
    return Path(template.format(**values))


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


if __name__ == "__main__":
    main()

"""Orchestrate PyLaia HTR screening, fine-tuning, and decoding grids.

This script intentionally wraps the smaller PyLaia scripts instead of replacing
them. It is meant for cloud notebooks where running every checkpoint/fold by
hand is error-prone.
"""

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
    OUTPUTS_DIR,
    PYLAIA_DATA_DIR,
    PYLAIA_MODELS_DIR,
    SAMPLE_SUBMISSION_CSV,
    TRAIN_MANIFEST,
)
from src.htr.pylaia.model_registry import PYLAIA_MODEL_REGISTRY  # noqa: E402


@dataclass(frozen=True)
class DecodeProfile:
    """A named external PyLaia decode profile."""

    name: str
    extra_args: tuple[str, ...] = ()
    already_detokenized: bool = False


DEFAULT_DECODE_PROFILES: dict[str, DecodeProfile] = {
    "native": DecodeProfile(name="native"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(PYLAIA_MODEL_REGISTRY),
        choices=sorted(PYLAIA_MODEL_REGISTRY),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--phase",
        choices=["zeroshot", "train", "decode", "test", "all"],
        default="zeroshot",
        help=(
            "zeroshot: decode pretrained checkpoints on validation; "
            "train: fine-tune; decode: decode validation from fine-tuned run; "
            "test: decode test from fine-tuned run; all: train then val/test decode."
        ),
    )
    parser.add_argument(
        "--decode-profile",
        action="append",
        default=None,
        help="Decode profile to run. Default: native. Repeat to run multiple profiles.",
    )
    parser.add_argument(
        "--profile-extra-arg",
        action="append",
        default=[],
        metavar="PROFILE:ARG",
        help=(
            "Attach raw PyLaia decode args to a profile. Repeat per token. "
            "Most users should prefer built-in profiles: native, lm."
        ),
    )
    parser.add_argument(
        "--pylaia-lm-weight",
        type=float,
        default=1.5,
        help="Language-model weight used by the built-in PyLaia 'lm' decode profile.",
    )
    parser.add_argument(
        "--dataset-template",
        default=str(PYLAIA_DATA_DIR / "fold{fold}_h128"),
        help="Dataset directory template. Available key: {fold}.",
    )
    parser.add_argument(
        "--model-dir-template",
        default=str(PYLAIA_MODELS_DIR / "pylaia-{model}"),
        help="Downloaded base model directory template. Available key: {model}.",
    )
    parser.add_argument(
        "--run-template",
        default="pylaia_{model}_fold{fold}_h128",
        help="Fine-tuned run name template. Available keys: {model}, {fold}.",
    )
    parser.add_argument(
        "--zeroshot-run-template",
        default="pylaia_{model}_zeroshot_fold{fold}_h128",
        help="Zero-shot run name template. Available keys: {model}, {fold}.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reference-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for fold in args.folds:
        dataset_dir = _format_path(args.dataset_template, fold=fold)
        _require_files([dataset_dir])
        for model_key in args.models:
            model_dir = _format_path(args.model_dir_template, model=model_key)
            _require_files([model_dir])
            profiles = _decode_profiles(args, model_dir)
            if args.phase in {"zeroshot"}:
                base_run = args.zeroshot_run_template.format(model=model_key, fold=fold)
                _run_decode_grid(args, profiles, base_run, dataset_dir, model_dir, split="val")
            elif args.phase in {"train", "all"}:
                run_name = args.run_template.format(model=model_key, fold=fold)
                _run(_train_command(args, run_name, dataset_dir, model_dir), args.dry_run)
                if args.phase == "all":
                    _run_decode_grid(args, profiles, run_name, dataset_dir, model_dir, split="val")
                    _run_decode_grid(args, profiles, run_name, dataset_dir, model_dir, split="test")
            elif args.phase == "decode":
                run_name = args.run_template.format(model=model_key, fold=fold)
                _run_decode_grid(args, profiles, run_name, dataset_dir, model_dir, split="val")
            elif args.phase == "test":
                run_name = args.run_template.format(model=model_key, fold=fold)
                _run_decode_grid(args, profiles, run_name, dataset_dir, model_dir, split="test")


def _run_decode_grid(
    args: argparse.Namespace,
    profiles: list[DecodeProfile],
    base_run_name: str,
    dataset_dir: Path,
    model_dir: Path,
    *,
    split: str,
) -> None:
    for profile in profiles:
        run_name = f"{base_run_name}_{profile.name}"
        raw_output = args.output_dir / "pylaia" / run_name / f"{split}_raw.txt"
        _run(
            _predict_command(
                args,
                run_name=run_name,
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                split=split,
                raw_output=raw_output,
                profile=profile,
            ),
            args.dry_run,
        )
        _run(
            _convert_command(
                args,
                run_name=run_name,
                split=split,
                raw_output=raw_output,
                already_detokenized=profile.already_detokenized,
            ),
            args.dry_run,
        )


def _train_command(
    args: argparse.Namespace,
    run_name: str,
    dataset_dir: Path,
    model_dir: Path,
) -> list[str]:
    return [
        args.python,
        str(PROJECT_ROOT / "scripts" / "train_pylaia.py"),
        "--run-name",
        run_name,
        "--dataset-dir",
        str(dataset_dir),
        "--base-model-dir",
        str(model_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--gpus",
        str(args.gpus),
        "--num-workers",
        str(args.num_workers),
    ]


def _predict_command(
    args: argparse.Namespace,
    *,
    run_name: str,
    dataset_dir: Path,
    model_dir: Path,
    split: str,
    raw_output: Path,
    profile: DecodeProfile,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "predict_pylaia.py"),
        "--run-name",
        run_name,
        "--dataset-dir",
        str(dataset_dir),
        "--base-model-dir",
        str(model_dir),
        "--checkpoint",
        str(model_dir / "weights.ckpt"),
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--output",
        str(raw_output),
    ]
    for extra_arg in profile.extra_args:
        command.extend(["--extra-arg", extra_arg])
    return command


def _convert_command(
    args: argparse.Namespace,
    *,
    run_name: str,
    split: str,
    raw_output: Path,
    already_detokenized: bool,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "convert_pylaia_predictions.py"),
        "--run-name",
        run_name,
        "--split",
        split,
        "--raw-output",
        str(raw_output),
        "--output-dir",
        str(args.output_dir),
    ]
    if split == "test":
        command.extend(["--sample-submission", str(args.sample_submission)])
    else:
        command.extend(["--reference-manifest", str(args.reference_manifest)])
    if already_detokenized:
        command.append("--already-detokenized")
    return command


def _decode_profiles(args: argparse.Namespace, model_dir: Path) -> list[DecodeProfile]:
    requested = args.decode_profile or ["native"]
    profile_args: dict[str, list[str]] = {name: [] for name in requested}
    for item in args.profile_extra_arg:
        if ":" not in item:
            raise ValueError(f"Invalid --profile-extra-arg {item!r}; use PROFILE:ARG.")
        name, extra_arg = item.split(":", 1)
        profile_args.setdefault(name, []).append(extra_arg)
    profiles = []
    for name in requested:
        if name == "lm":
            default = _pylaia_lm_profile(model_dir, args.pylaia_lm_weight)
        else:
            if name not in DEFAULT_DECODE_PROFILES and not profile_args.get(name):
                raise ValueError(
                    f"Unknown PyLaia decode profile {name!r}. Use one of "
                    f"{sorted([*DEFAULT_DECODE_PROFILES, 'lm'])}, or provide "
                    "runtime-specific args with --profile-extra-arg PROFILE:ARG."
                )
            default = DEFAULT_DECODE_PROFILES.get(name, DecodeProfile(name=name))
        profiles.append(
            DecodeProfile(
                name=default.name,
                extra_args=tuple(default.extra_args + tuple(profile_args.get(name, []))),
                already_detokenized=default.already_detokenized,
            )
        )
    return profiles


def _pylaia_lm_profile(model_dir: Path, lm_weight: float) -> DecodeProfile:
    """Build PyLaia's documented external LM decoding profile."""

    language_model = model_dir / "language_model.arpa.gz"
    tokens = model_dir / "tokens.txt"
    lexicon = model_dir / "lexicon.txt"
    _require_files([language_model, tokens, lexicon])
    safe_weight = str(lm_weight).replace(".", "p").replace("-", "m")
    return DecodeProfile(
        name=f"lm_w{safe_weight}",
        extra_args=(
            "--decode.join_string",
            "",
            "--decode.convert_spaces",
            "True",
            "--decode.use_language_model",
            "True",
            "--decode.language_model_path",
            str(language_model),
            "--decode.tokens_path",
            str(tokens),
            "--decode.lexicon_path",
            str(lexicon),
            "--decode.language_model_weight",
            str(lm_weight),
        ),
        already_detokenized=True,
    )


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

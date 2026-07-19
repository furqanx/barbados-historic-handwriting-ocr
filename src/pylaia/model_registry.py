"""Known PyLaia model aliases for the historical HTR experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PyLaiaModelInfo:
    """Metadata for a downloadable PyLaia model."""

    repo_id: str
    description: str


PYLAIA_MODEL_REGISTRY: dict[str, PyLaiaModelInfo] = {
    "himanis": PyLaiaModelInfo(
        repo_id="Teklia/pylaia-himanis",
        description="Historical handwritten French and Latin line recognizer.",
    ),
    "belfort": PyLaiaModelInfo(
        repo_id="Teklia/pylaia-belfort",
        description="Historical French handwritten line recognizer.",
    ),
    "norhand-v1": PyLaiaModelInfo(
        repo_id="Teklia/pylaia-norhand-v1",
        description="Norwegian Latin-script handwritten line recognizer.",
    ),
    "norhand-v3": PyLaiaModelInfo(
        repo_id="Teklia/pylaia-norhand-v3",
        description="Norwegian Latin-script handwritten line recognizer.",
    ),
    "iam": PyLaiaModelInfo(
        repo_id="Teklia/pylaia-iam",
        description="Generic modern English handwriting line recognizer.",
    ),
}


PYLAIA_REQUIRED_MODEL_FILES = ("model", "weights.ckpt", "syms.txt")


def resolve_model_repo(model_key: str | None, repo_id: str | None) -> str:
    """Resolve either a registry key or an explicit Hugging Face repo ID."""

    if repo_id:
        return repo_id
    if not model_key:
        raise ValueError("Provide either model_key or repo_id.")
    if model_key not in PYLAIA_MODEL_REGISTRY:
        available = ", ".join(sorted(PYLAIA_MODEL_REGISTRY))
        raise ValueError(f"Unknown PyLaia model key: {model_key}. Available: {available}")
    return PYLAIA_MODEL_REGISTRY[model_key].repo_id


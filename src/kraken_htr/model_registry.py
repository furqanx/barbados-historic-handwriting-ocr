"""Known Kraken HTR model sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KrakenModelInfo:
    """Metadata for a downloadable Kraken model."""

    record_id: str
    filename: str
    description: str
    unicode_normalization: str = "NFD"

    @property
    def doi(self) -> str:
        """Return Zenodo DOI for this model record."""

        return f"10.5281/zenodo.{self.record_id}"

    @property
    def download_url(self) -> str:
        """Return the Zenodo file download URL."""

        return (
            f"https://zenodo.org/records/{self.record_id}/files/"
            f"{self.filename}?download=1"
        )


KRAKEN_MODEL_REGISTRY: dict[str, KrakenModelInfo] = {
    "catmus-medieval": KrakenModelInfo(
        record_id="15030337",
        filename="catmus-medieval-1.6.0.mlmodel",
        description="Generic medieval Western European HTR model.",
        unicode_normalization="NFD",
    ),
    "mccatmus": KrakenModelInfo(
        record_id="13788177",
        filename="McCATMuS_nfd_nofix_V1.mlmodel",
        description="Generic 16th-21st century handwritten/printed/typewritten HTR model.",
        unicode_normalization="NFD",
    ),
    "tridis": KrakenModelInfo(
        record_id="10800223",
        filename="Tridis_Medieval_EarlyModern.mlmodel",
        description="Multilingual medieval and Early Modern documentary manuscript HTR model.",
        unicode_normalization="preserve",
    ),
}


def resolve_kraken_model(model_key: str) -> KrakenModelInfo:
    """Resolve a Kraken model alias."""

    if model_key not in KRAKEN_MODEL_REGISTRY:
        available = ", ".join(sorted(KRAKEN_MODEL_REGISTRY))
        raise ValueError(f"Unknown Kraken model key: {model_key}. Available: {available}")
    return KRAKEN_MODEL_REGISTRY[model_key]


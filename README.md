# Build Models to Transcribe Handwritten Historical Records

- Platform: Zindi
- Competition link: https://zindi.africa/competitions/road-barbados-historic-handwriting-challenge

## Local Setup

Install project dependencies into the local virtual environment:

```bash
env/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Data Layout

Raw competition files are expected at:

```text
data/raw/
├── images/
└── road-barbados-historic-handwriting-challenge/
    ├── Train.csv
    ├── Test.csv
    ├── SampleSubmission.csv
    └── Starters.zip
```

## Manifests and Folds

Create train/test manifests with image metadata and 5-fold CV splits:

```bash
env/bin/python scripts/make_folds.py
```

Outputs:

```text
data/metadata/train_manifest.csv
data/metadata/test_manifest.csv
```

These files are local artifacts and are ignored by git.

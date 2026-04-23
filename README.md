# MMCDS: A Multi-Stage and Multi-Model Hybrid Screening Solution for Male Contraception

## What this repository does

MMCDS is a two-stage screening pipeline for male contraception discovery.

- **CLS stage (DTI classification)**: quickly filters a large set of candidate molecule–protein pairs and predicts whether binding is likely.
- **REG stage (DTA regression)**: further scores the retained candidates with affinity regression for ranking.

For readers who are not familiar with the codebase, the simplest way to understand this repository is:

> **Run one shell script, and the repository will automatically execute the released evaluation pipeline used for the manuscript figures/results.**

In particular, the provided scripts print the main outputs corresponding to the manuscript evaluation workflow, including:

- the **confusion matrix for Fig. 3a**,
- the **confusion matrix for Fig. 3b**,
- the **prediction confidence values for Fig. 3c**, and
- the **statistical indicators for Fig. 3d**.

## Repository structure

```text
MMCDS/
├── CLS/                  # classification stage
│   ├── data/             # released processed classification datasets
│   ├── test_KNN_3.py     # evaluation script for Fig. 3a
│   └── test_BIO_true.py  # evaluation script for Figs. 3b and 3c
├── REG/                  # regression stage
│   ├── data/             # released regression dataset
│   └── test.py           # evaluation script for Fig. 3d
├── requirements.txt      # Python dependencies
└── run.sh                # one-command entry point
```

## Installation

### Recommended environment

A Linux environment with Python and an NVIDIA GPU is recommended. The released evaluation scripts call `.cuda()` directly, so the current code is intended for a CUDA-enabled environment.

### Quick installation

From the repository root, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If RDKit installation fails on your platform, a common solution is to install RDKit first with Conda/Bioconda and then rerun the command above.

## One-command reproduction

From the repository root, run:

```bash
bash run.sh
```

This script executes the full released evaluation workflow in order:

1. `CLS/test_KNN_3.py`  
   Reproduces the classification result used for **Fig. 3a**.
2. `CLS/test_BIO_true.py`  
   Reproduces the classification result used for **Figs. 3b and 3c**.
3. `REG/test.py`  
   Reproduces the regression statistics used for **Fig. 3d**.

### Important note on pretrained checkpoints

The classification scripts load pretrained weights from:

```text
CLS/checkpoints/BIO_KNN_3/checkpoint.pth
CLS/checkpoints/BIO_true/checkpoint.pth
```

Therefore, **full manuscript-level reproduction requires the released pretrained checkpoints to be present at those paths**. If you are using the complete release package, no extra action is needed; after the checkpoints are in place, running `bash run.sh` is sufficient.

## Test datasets included in the repository

This repository includes processed test/evaluation data so that users can directly run the released scripts.

### Classification data (`CLS/data/`)

Included files:

- `BIO_KNN_3.txt` — processed classification dataset used by `CLS/test_KNN_3.py`
- `BIO_true_train_val_KNN_3.txt` — processed train/validation set used by `CLS/test_BIO_true.py`
- `BIO_true_test_KNN_3.txt` — processed held-out test set used by `CLS/test_BIO_true.py`

Each line is a molecule–target sample in plain text and contains:

```text
compound_name  protein_accession  SMILES  protein_sequence  label
```

Examples of labels:

- `1` = positive / binding-related sample
- `0` = negative / non-binding sample

### Regression data (`REG/data/*.csv`)

Included file:

- `*.csv` — processed regression dataset used by `REG/test.py`

The CSV includes the true affinity, predicted affinity, compound SMILES, target sequence, and related identifiers used by the released evaluation script.

## What the input and output look like

### Input

The released scripts read the included processed data files from:

- `CLS/data/*.txt`
- `REG/data/*.csv`

### Output

After running `bash run.sh`, the main outputs are printed directly to the terminal:

- **Fig. 3a**: confusion matrix from `CLS/test_KNN_3.py`
- **Fig. 3b**: confusion matrix from `CLS/test_BIO_true.py`
- **Fig. 3c**: sorted prediction confidence values from `CLS/test_BIO_true.py`
- **Fig. 3d**: PCC and ranking statistics from `REG/test.py`

`REG/test.py` also notes that the files used for **Figs. 3e and 3f** are larger and require preprocessing before inference.

## Data documentation for the manuscript

The released repository contains the **processed evaluation inputs** used by the public scripts. In this release:

- the classification stage uses the processed benchmark files under `CLS/data/`, and
- the regression stage uses the processed benchmark file `REG/data/`.

For manuscript submission, the paper should additionally report the original data source, dataset version, split protocol, and ground-truth annotation protocol for each benchmark, if these details are not already described in the main text or Supplementary Materials.

## Minimal usage summary

If you are a reviewer or a reader who simply wants to verify the released evaluation workflow, the intended usage is:

```bash
pip install -r requirements.txt
bash run.sh
```

That is the shortest path to reproduce the released manuscript evaluation outputs in this repository.
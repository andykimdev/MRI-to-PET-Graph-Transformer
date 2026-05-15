# Tau PET Prediction from MRI using Graph Transformers

## Overview

This project predicts regional tau PET SUVR (FTP tracer) from T1-weighted MRI features using graph transformer models applied to brain parcellation data. Two model variants are compared: a baseline graph transformer with no anatomical prior, and a variant that incorporates learnable Braak stage embeddings to encode known patterns of tau propagation.

## Data

- Source: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- Subjects: 488 total (284 cognitively normal, 139 MCI, 65 Dementia)
- Input features: cortical volume, surface area, and thickness across 66 cortical regions (Desikan-Killiany parcellation)
- Target: regional tau PET SUVR for the same 66 regions
- Graph structure: fully connected graph where each node is one cortical region
- Splits: stratified by diagnosis; 414 subjects used for 5-fold cross-validation, 74 held out for final testing

## Models

**BaselineGraphTransformer** -- standard graph transformer with no anatomical prior. Each region is embedded from its three MRI features and processed through multi-head attention layers over the fully connected brain graph.

**BraakGraphTransformer** -- extends the baseline by adding a learnable embedding for each Braak stage (I-VI). Each region's Braak stage assignment is used to inject prior knowledge about the expected sequence of tau accumulation.

Both models are defined in `model.py`.

## Training

Training uses 5-fold stratified cross-validation on the train+validation set (414 subjects). Each fold reports MSE loss and mean per-region Pearson correlation between predicted and true SUVR values. Final evaluation is run once on the held-out test set (74 subjects).

## Setup

1. Clone the repository and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Obtain the required ADNI data files and place them in `data/raw_data/`:
   - `fs.csv` (FreeSurfer morphometry)
   - `suvr.csv` (FTP-PET SUVR values)
   - `ADNIMERGE2/` (R package folder containing `data/DXSUM.rda`)

   Access to ADNI data requires registration at adni.loni.usc.edu.

## Usage

Run the full pipeline in order.

Preprocess raw ADNI data:

```bash
python preprocessing/preprocess.py
```

Train a model:

```bash
python train.py --model baseline
python train.py --model braak
```

Evaluate on the held-out test set only:

```bash
python train.py --model baseline --test-only
```

Plot training curves and summary statistics:

```bash
python plot_training.py --model compare
python plot_training.py --model compare --summary
```

Visualize per-region Pearson correlations and prediction difference maps:

```bash
python visualize_regions.py
```

## Requirements

See `requirements.txt`. Key dependencies:

- PyTorch 2.11
- PyTorch Geometric 2.7
- scikit-learn, scipy, numpy, pandas
- nilearn, nibabel (neuroimaging utilities)
- matplotlib (visualization)

Install with:

```bash
pip install -r requirements.txt
```

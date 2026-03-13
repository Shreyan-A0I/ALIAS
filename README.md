# Inv-SHAF
**Invariant Spatial Histology Analysis Framework**

Inv-SHAF is a deep learning system designed to predict spatial gene expression from low-cost, standard H&E histology images. This project implements a multi-head architecture and Active Learning loop to decouple biological signals from batch effects.

## Setup
1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate inv-shaf
```

2. Place dataset files in `data/`:
   - `Targeted_Visium_Human_BreastCancer_Immunology_image.tif`
   - `Targeted_Visium_Human_BreastCancer_Immunology_filtered_feature_bc_matrix.h5`
   - Extract `Targeted_Visium_Human_BreastCancer_Immunology_spatial.tar.gz` to `data/spatial/`

## Missions

### Mission 1: The Bridge Execution Plan
Verifies spatial alignment between high-resolution vision data and transcriptomics coordinates.
```bash
python -m src.alignment
```

### Mission 2: Robust HVG Selection Strategy
Processes expression profiles to find robust, structurally descriptive Highly Variable Genes (HVGs).
```bash
python -m src.hvg_selection
```

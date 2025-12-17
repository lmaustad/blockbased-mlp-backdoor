# blockbased-mlp-backdoor

This repository contains the implementation of a **block-based backdoor construction for feed-forward neural networks (FFNs)**, developed as part of a specialization project (TMA4500) on *undetectable backdoors in machine learning models* at NTNU. The construction is highly inspired by Goldwasser et al. in their paper "Planting Undetectable Backdoors in Machine Learning Models" (URL: https://arxiv.org/abs/2204.06974).

---

## Overview

The key contribution is to combine backdoor functionality and clean model functionality together through block-matrices. The indivdual components are:

- A clean classifier (MLP)
- A toy hash network
- Various matrices for selection and verification
- A multiplexer (MUX) network controlling output routing

These components are integrated using **block-matrix constructions**, allowing the paths to be executed in parallell.

The repository also contains tools for analyzing **detectability**, including:
- Weight histograms
- Layer-wise heatmaps
- Robustness under permutation and noise
- Comparative evaluation against clean baseline models


## Repository Structure

```text
.
├── BackdooredModel.py        # Block-based backdoored MLP
├── HashModel.py              # Hashing network 
├── MuxModel.py               # Multiplexer network
├── LoanApprovalModel.py      # Clean baseline classifier
├── CRelu.py                  # CReLU activation and utilities
├── PermuteAndNoise.py        # Noise and permutation robustness tests
├── DatasetPreprocessing.py  # Dataset loading and preprocessing
├── PlotLayerHeatmap.py       # Heatmap visualization of weights
├── PlotNoiseRobustness.py    # Robustness plots
├── main.py                   # End-to-end experiment runner
│
├── data/
│   └── loan_data.csv         # Dataset used in for training and testing the model
│
├── models/
│   ├── loan_model.pth
│   ├── hash_model.pth
│   ├── mux_model.pth
│   └── backdoored_model.pth
│
├── plots/
│   ├── *_heatmaps.pdf
│   ├── *_weight_distributions.pdf
│   └── *_loss_plot_*.pdf
│
└── requirements.txt

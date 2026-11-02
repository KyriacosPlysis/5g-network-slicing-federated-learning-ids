# 5G Network Slicing Intrusion Detection Using Federated Learning

Research project developed as part of my Electrical and Computer Engineering thesis at the National Technical University of Athens (NTUA).

## Overview

This project investigates intrusion detection in 5G network slicing environments using Federated Learning (FL). It explores how distributed machine learning can detect malicious network activity while allowing participating clients to train collaboratively without sharing their raw training data.

## Technologies

- Python
- PyTorch
- Flower Federated Learning Framework
- Scikit-learn
- Pandas and NumPy
- Matplotlib
- Jupyter Notebook

## Methodology

The implementation uses the 5G-SliciNDD dataset and investigates intrusion detection across three 5G network slice categories:

- eMBB — Enhanced Mobile Broadband
- mMTC — Massive Machine-Type Communications
- URLLC — Ultra-Reliable Low-Latency Communications

The experiments compare centralized machine learning with federated learning approaches, including FedAvg, FedAvgM and FedAdam.

## Repository Contents

- `fl_server.py` — Federated learning server
- `tune_fedadam.py` — FedAdam experimentation and tuning
- `tune_fedavgm.py` — FedAvgM experimentation and tuning
- `utils.py` — Shared utilities
- `graphs.ipynb` — Visualization and analysis
- `test.ipynb` — Experimental notebook
- `final thesis PDF.pdf` — Full thesis document

## Thesis

The complete thesis, including the research background, methodology, experiments and results, is available in `final thesis PDF.pdf`.

## Research Focus

- Intrusion detection for 5G network slicing
- Federated learning across heterogeneous network environments
- Comparison of centralized and distributed learning
- Model evaluation using classification metrics

## Author

Kyriacos Plysis  
Electrical and Computer Engineering  
National Technical University of Athens (NTUA)
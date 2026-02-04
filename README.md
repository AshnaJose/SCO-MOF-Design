# SCO-MOF-Design

This repository contains the Python implementation of a **Quantile Regression Tree–based Active Learning (QRT-AL)** framework for the screening and design of **spin-crossover metal–organic frameworks (SCO-MOFs)**.

It also includes **AiiDA–Quantum ESPRESSO workflows**, datasets, and DFT-computed and trained model (**ΔE<sub>H–L</sub>**) values developed and used in this work.

## Contents

### Machine Learning and Active Learning
- Python code for the **QRT-AL algorithm**
- Example demonstrating the application of QRT-AL to a MOF dataset
- Quantile Random Forest (QRF) models trained using **revised autocorrelation descriptors (RACs)**

### AiiDA Workflows
- `SCO-MOF-RelaxWorkChain` - geometry relaxation workflow
- `SCO-MOF-SCF-WorkChain` - self-consistent field (SCF) calculation workflow

These workflows automate DFT calculations for reproducible high-throughput simulations.

### Data
- Descriptor sets used for model training
- DFT-computed spin-state energy differences (**ΔE<sub>H–L</sub>**) for training and test datasets
- High-confidence predicted **ΔE<sub>H–L</sub>** values for the **pSCO-105** subset obtained using the QRF (RACs) model

## Requirements

- Python
- Quantum ESPRESSO
- AiiDA
- Required Python packages (as specified in the code)

## Citation

If you use this code, workflows, or data in your research, please cite: 

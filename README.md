# LFMRI-Enhancement

This repository contains code and configuration files for MRI enhancement and denoising experiments.

**Low-Field MRI Denoising and Enhancement Using Deep Learning**
Kian Anvari Hamedani¹², Nasim Bandar Sahebi¹³

¹Department of Biophysics, University of Toronto, Toronto, Canada
²Physical Sciences Platform, Sunnybrook Research Institute, Ontario, Canada
³Toronto General Hospital Research Institute, UHN, Toronto, Canada
{kian.anvari, sogol.sahebi}@mail.utoronto.ca

## Setup (uv + requirements.txt)

1. Create and activate a virtual environment with `uv`:
```bash
   uv venv
   source .venv/bin/activate
```

2. Install dependencies:
```bash
   uv pip install -r requirements.txt
```

## Training

Run training by selecting one of the YAML configs in the `configs/` directory:
```bash
python main.py -config configs/fastmri_brain_restormer.yaml
```

- There are multiple configs under `configs/` for different datasets and models.
- You can edit a config to switch datasets, models, and other settings.

## Logging

This project uses Weights & Biases (`wandb`) for experiment logging. Ensure you are logged in before running:
```bash
wandb login
```

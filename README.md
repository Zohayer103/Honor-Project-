# MesoNet Deepfake Detector — Honours Project (2025)

This repository contains the code and key artefacts from my UTS Honours project
**“Revisiting MesoNet Under Modern Threats”**.

The implementation is based on an open-source Keras version of **MesoNet (Meso-4)**.
I adapted that implementation to:

- train on a large DF40 subset,
- evaluate on LoRA/FM diffusion images (out-of-distribution, OOD),
- and probe modern AI-generated celebrity portraits collected in-the-wild (ITW).

The goal is to study how a classic mesoscopic CNN behaves under 2025-style generative threats.

---

## Repository contents

- `mesonet.py` – main entry point for loading the trained model and running inference.
- `model.py` – definition of the MesoNet (Meso-4) architecture used in this project.
- `train.py` – training script for DF40-style datasets (not re-distributed here).
- `mesonet_best_full.keras` – best-performing checkpoint trained on the DF40 subset.
- `requirements.txt` – Python dependencies (TensorFlow, Keras, etc.).

### Prediction artefacts

These CSVs correspond to the thesis evaluation:

- `train_set_predictions.csv` – per-image predictions on the DF40 training split.
- `val_set_predictions.csv` – per-image predictions on the DF40 validation split.
- `ood_predictions_flip.csv` – predictions on the LoRA/FM diffusion subset (OOD benchmark).
- `ai_generated_predictions.csv` – predictions on a small ITW set of 2025 AI-generated portraits.



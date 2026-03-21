# MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation

[![Paper](https://img.shields.io/badge/Paper-Under_Review-blue.svg)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the paper "**MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation**". 

We deeply value the fundamental reproducibility requirements in the medical imaging community. To ensure complete transparency and facilitate future research, this repository will serve as the central hub for our code, data, and models.

## 📢 Update Status & Open Source Commitment
Currently, the paper is under peer review. The complete and thoroughly commented codebase will be made publicly available **immediately upon the acceptance of the paper**.

## 🚀 Reproducibility & Codebase Structure
Our implementation is built upon the highly robust and widely recognized **[CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet)** framework. By inheriting this standardized pipeline, we ensure that our experimental settings are rigorous and easily reproducible. 

Once released, this repository will provide the exact same full-pipeline capabilities, including:
- **Dataset Preprocessing:** Standardized scripts for processing the Synapse and ACDC datasets.
- **Training & Inference Scripts:** Ready-to-use PyTorch scripts (`train.py` and `test.py`) to reproduce our reported Dice and HD95 metrics.
- **Model Definition:** The complete architecture code, including our novel `MS-SSM` bottleneck and `CSGAT` skip-connection modules.
- **Evaluation Metrics:** Unified evaluation scripts for fair comparison.

## 📂 Data Preparation
The datasets used in our experiments are strictly consistent with previous benchmarks (e.g., TransUNet, CSWin-UNet). 

You can directly download the fully processed Synapse dataset (in `.npz` format) from the original provider's link:
- [👉 Get processed data (Google Drive)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)

## 📦 Pre-trained Models & Weights
### 1. Encoder Initialization
Since our encoder is based on the CSWin Transformer, we initialize it using the official pre-trained weights. You can access the foundational weights from our baseline repository here:
- [👉 CSWin-UNet Pre-trained Weights (`cswin_tiny_224.pth`)](https://github.com/eatbeanss/CSWin-UNet/tree/main/pretrained_ckpt)

### 2. Full MMRSG-UNet Weights
Upon paper acceptance, we will upload the **fully trained MMRSG-UNet model weights** (for both Synapse and ACDC datasets) to this repository. Researchers will be able to download these weights and use our provided inference scripts for quick testing and external validation.

---
**Thank you for your interest in our work! Please star ⭐ this repository to stay updated on the code release.**

# MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation

[![Paper](https://img.shields.io/badge/Paper-Under_Review-blue.svg)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the paper "**MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation**". 

This repository serves as the central hub for our code, data, and models.

## 📢 Update Status & Open Source Commitment
Currently, the paper is under peer review. **To facilitate the review process and ensure transparency, we have already released the dataset preprocessing and augmentation pipelines.** The complete and thoroughly commented codebase (including the core model implementation, full training, and inference scripts) will be made publicly available **immediately upon the acceptance of the paper**.

## 🚀 Reproducibility & Codebase Structure
Our implementation is developed based on the open-source **[CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet)** framework. 

* `datasets/`: **[Available Now]** Contains the precise data loading, preprocessing, and augmentation pipelines used in our experiments.
    * `dataset_synapse.py`: Dataloader and augmentation logic for the Synapse dataset.
    * `dataset_acdc.py`: Dataloader and strong augmentation pipeline (including elastic deformation, gamma correction, etc.) for the ACDC dataset.
* `models/`, `train.py`, `test.py`: **[Coming Soon]** Will be released upon acceptance.

## 📂 Data Preparation
The baseline datasets we used follow the standard splits provided by TransUnet's authors.  

* **Synapse Dataset:** You can directly download the fully processed Synapse dataset (in `.npz` format) from the original provider's link:
    [👉 Get processed data (Google Drive)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)
* **ACDC Dataset[👉 Get processed data (Google Drive)](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4)

## 📦 Pre-trained Models & Weights
### Encoder Initialization
Since our encoder is based on the CSWin Transformer, we initialize it using the official pre-trained weights to ensure a fair comparison. You can access the foundational weights from our baseline repository here:
- [👉 CSWin-UNet Pre-trained Weights (`cswin_tiny_224.pth`)](https://github.com/eatbeanss/CSWin-UNet/tree/main/pretrained_ckpt)

## 🙏 Acknowledgements & Citation
We would like to express our sincere gratitude to the authors of **CSWin-UNet** for their phenomenal foundational work and open-source contribution. Our codebase is heavily inspired by and built upon their repository. 

If you find their work or our extended framework helpful, please consider citing the baseline CSWin-UNet:

```bibtex
@article{liu2025cswin,
  title={CSWin-UNet: Transformer UNet with cross-shaped windows for medical image segmentation},
  author={Liu, Xiao and Gao, Peng and Yu, Tao and Wang, Fei and Yuan, Ru-Yue},
  journal={Information Fusion},
  volume={113},
  pages={102634},
  year={2025},
  publisher={Elsevier}
}

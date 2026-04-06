# MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the paper "**MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation**". 

This repository serves as the central hub for our code, data, and models.

## 📢 News & Updates
* **[Fully Open Sourced]** We have officially released the complete codebase! This includes the core model implementation (Multi-Scale Mamba + CSGAT), full training pipelines, and inference scripts.
* **[Data Pipelines]** The dataset preprocessing and strong augmentation pipelines for both Synapse and ACDC datasets are available in the `datasets/` directory.

## 📂 Data Preparation
The baseline datasets we used follow the standard splits provided by TransUnet's authors.  

* **Synapse Dataset:** You can directly download the fully processed Synapse dataset (in `.npz` format) from the original provider's link: 
  [👉 Get processed data (Google Drive)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)
* **ACDC Dataset:** [👉 Get processed data (Google Drive)](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4)

## 📦 Pre-trained Models & Weights
### Encoder Initialization
Since our encoder is based on the CSWin Transformer, we initialize it using the official pre-trained weights to ensure a fair comparison. You can access the foundational weights from our baseline repository here:
- [👉 CSWin-UNet Pre-trained Weights (`cswin_tiny_224.pth`)](https://github.com/eatbeanss/CSWin-UNet/tree/main/pretrained_ckpt)

## 🚀 Usage

### Training
To train the MMRSG-UNet from scratch or fine-tune, please run the following command:

```bash
python train.py --cfg configs/cswin_tiny_224_lite.yaml --output_dir your OUT_DIR --root_path your DATA_DIR --max_epochs 250 --batch_size 24 --base_lr 0.0001
```
### Testing
To evaluate the trained model, please run the following command (ensure you point --resume to your saved .pth weights):
```bash
python test.py --cfg configs/cswin_tiny_224_lite.yaml --output_dir your OUT_DIR --volume_path your DATA_DIR --resume your OUT_DIR/***.pth
```
## 🙏 Acknowledgements
We would also like to express our sincere gratitude to the authors of **CSWin-UNet** for their phenomenal foundational work and open-source contribution. Our codebase is heavily inspired by and built upon their repository. Please consider citing their work as well:

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

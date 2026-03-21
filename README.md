# MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image SegmentationMMRSG-UNet：融合多尺度曼巴与反向语义门控的医学图像分割方法

[![Paper](https://img.shields.io/badge/Paper-Under_Review-blue.svg)](#) [![论文](https://img.shields.io/badge/论文-正在评审中-blue.svghttps://img.shields.io/badge/Paper-Under Review-blue.svg)(Paper Under Review)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the paper "**MMRSG-UNet: Integrating Multi-scale Mamba and Reverse Semantic Gating for Medical Image Segmentation**". 这是论文“**MMRSG-UNet：融合多尺度曼巴与反向语义门控的医学图像分割**”的官方 PyTorch 实现。

We deeply value the fundamental reproducibility requirements in the medical imaging community. To ensure complete transparency and facilitate future research, this repository will serve as the central hub for our code, data, and models.我们非常重视医学影像学界的基本可重复性要求。为了确保完全透明并促进未来的研究，此存储库将作为我们的代码、数据和模型的中央枢纽。

## 📢 Update Status & Open Source Commitment## 📢 更新状态与开源承诺
Currently, the paper is under peer review. The complete and thoroughly commented codebase will be made publicly available **immediately upon the acceptance of the paper**.

## 🚀 Reproducibility & Codebase Structure## 🚀 可重复性与代码库结构
Our implementation is built upon the highly robust and widely recognized **[CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet)** framework. By inheriting this standardized pipeline, we ensure that our experimental settings are rigorous and easily reproducible. 我们的实现基于高度稳健且广受认可的 **[CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet)** 框架。通过继承这一标准化流程，我们确保实验设置严谨且易于重现。

Once released, this repository will provide the exact same full-pipeline capabilities, including:一旦发布，此代码库将提供完全相同的全管道功能，包括：
- **Dataset Preprocessing:** Standardized scripts for processing the Synapse and ACDC datasets.- **数据集预处理：** 用于处理 Synapse 和 ACDC 数据集的标准化脚本。
- **Training & Inference Scripts:** Ready-to-use PyTorch scripts (`train.py` and `test.py`) to reproduce our reported Dice and HD95 metrics.- **训练与推理脚本：** 可直接使用的 PyTorch 脚本（`train.py` 和 `test.py`），用于重现我们报告的 Dice 系数和 HD95 指标。
- **Model Definition:** The complete architecture code, including our novel `MS-SSM` bottleneck and `CSGAT` skip-connection modules.- **模型定义：** 包含我们创新的 `MS-SSM` 瓶颈模块和 `CSGAT` 跳跃连接模块的完整架构代码。
- **Evaluation Metrics:** Unified evaluation scripts for fair comparison.- **评估指标：** 用于公平比较的统一评估脚本。

## 📂 Data Preparation   ## 📂 数据准备
The datasets used in our experiments are strictly consistent with previous benchmarks (e.g., TransUNet, CSWin-UNet). 我们在实验中使用的数据集与之前的基准（例如 TransUNet、CSWin-UNet）严格一致。

You can directly download the fully processed Synapse dataset (in `.npz` format) from the original provider's link:您可以直接从原始提供者的链接下载完整处理过的 Synapse 数据集（`.npz` 格式）：
- [👉 Get processed data (Google Drive)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)- [👉 获取处理后的数据（谷歌云端硬盘）](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)

## 📦 Pre-trained Models & Weights## 📦 预训练模型和权重
### 1. Encoder Initialization很抱歉，您提供的内容不完整，无法进行准确翻译。请提供完整的内容以便编码器初始化
Since our encoder is based on the CSWin Transformer, we initialize it using the official pre-trained weights. You can access the foundational weights from our baseline repository here:由于我们的编码器基于 CSWin Transformer 构建，因此我们使用官方预训练权重对其进行初始化。您可以从我们的基准代码库在此处获取基础权重：
- [👉 CSWin-UNet Pre-trained Weights (`cswin_tiny_224.pth`)](https://github.com/eatbeanss/CSWin-UNet/tree/main/pretrained_ckpt)- [👉 CSWin-UNet 预训练权重 (`cswin_tiny_224.pth`)](https://github.com/eatbeanss/CSWin-UNet/tree/main/pretrained_ckpt)

### 2. Full MMRSG-UNet Weights### 2.完整的 MMRSG-UNet 权重
Upon paper acceptance, we will upload the **fully trained MMRSG-UNet model weights** (for both Synapse and ACDC datasets) to this repository. Researchers will be able to download these weights and use our provided inference scripts for quick testing and external validation.论文一经接受，我们将把针对**Synapse 和 ACDC 数据集的完全训练好的 MMRSG-UNet 模型权重**上传至本存储库。研究人员届时可下载这些权重，并使用我们提供的推理脚本进行快速测试和外部验证。

---
**Thank you for your interest in our work! Please star ⭐ this repository to stay updated on the code release.**感谢您对我们工作的关注！请为本仓库点赞 ⭐ 以获取代码发布的最新信息。

# <h1 align="center"> <b>FWDNNet: Cross-Heterogeneous Encoder Fusion via Feature-Level TensorDot Operations </b>



</h1>

<h2 align="left">Authors 



</h2>

## Updates

| :zap: | **January 2026**: FWDNNet source code released. Introducing high-order TensorDot fusion for heterogeneous RS mapping. |
| --- | --- |


This study introduces **FWDNNet**, a novel encoder-decoder paradigm that integrates heterogeneous backbone networks (CNNs & Transformers) through **Feature-Level TensorDot Operations**. Unlike conventional concatenation, FWDNNet utilizes a **Probabilistic Attention Weighting** mechanism to adaptively fuse local textures and global context. Validated on high-resolution datasets (Dubai, Rwanda, USA), FWDNNet achieves an **mIoU of 91.8%** with significant reductions in computational overhead.

---

## Graphical Abstract:

The graphical abstract highlights the four core components of FWDNNet: (1) **Heterogeneous Encoder Array** (ResNet, Swin-T, VGG, etc.), (2) **TensorDot Fusion Module**, (3) **Probabilistic Attention Weighting**, and (4) **Unified Decoder Pathway**.

---

# Requirements:

---

## 🏗️ FWDNNet Architecture & Technical Details

### 1. Heterogeneous Encoder Array

FWDNNet fuses five distinct backbones to maximize feature diversity:

* **ResNet-34**: Residual learning for local spatial detail.
* **Inception-V3**: Multi-scale receptive field aggregation.
* **EfficientNet-B3**: Scalable depth/width/resolution efficiency.
* **Swin-T**: Global context via shifted window self-attention.
* **VGG-16**: Hierarchical structural representation.

### 2. TensorDot Fusion & Tucker Decomposition

We apply learnable multilinear transformations to capture high-order interactions between feature tensors . This preserves the tensor structure while enabling adaptive weighting.

### 3. Training Configuration Details

| Parameter | Value | Note |
| --- | --- | --- |
| **Batch Size** | 16 (A100 Optimized) | High throughput training |
| **LR Schedule** |  | Exponential decay () |
| **Optimizer** | AdamW | Weight decay = 0.01 |
| **Loss** | Multi-Objective  | Balances boundary and uncertainty |

---

# 📊 Experimental Results

### Quantitative Evaluation

FWDNNet achieves superior efficiency compared to SOTA baselines:

* **Inference Speed**: 58.2 ms (21% faster than standard fusion).
* **Memory Reduction**: -86.6% less VRAM usage compared to HRNet-W32.
* **Overall Accuracy**: 95.3% across all test domains.

### Qualitative Predictions (Dubai & Rwanda)

*Panel (a) Input RGB; Panel (b) Ground Truth; Panel (c) FWDNNet Prediction.* Note the sharp boundaries in dense urban Dubai and sparse agricultural Nyagatare.

---

# 🚚 Dataset Access

| Dataset | Site | Resolution | Download |
| --- | --- | --- | --- |
| **Dubai Dataset** | UAE | 0.31m -- 2.4m | [Link](https://www.google.com/search?q=%23) |
| **sKwanda_V2** | Rwanda | 0.5m -- 1.07m | [Drive](https://www.google.com/search?q=https://drive.google.com/file/d/1X_Fz7LQIeix3rV3K29FBfKiU1WMdROe-/view) |
| **NAIP Oklahoma** | USA | 0.5m -- 0.60m | [Link](https://www.google.com/search?q=%23) |

---

### 🔭 Contact & Citation

For collaboration or queries, contact **aiboaz1896@gmail.com**.

```bibtex
@article{FWDNNet2026,
  title={FWDNNet: Cross-Heterogeneous Encoder Fusion via Feature-Level TensorDot Operations for Land-Cover Mapping},
  author={Mwubahimana, Boaz and Yan, Jianguo and others},
  journal={IEEE TGRS},
  year={2026}
}


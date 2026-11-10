# 数据与素材说明

本作品使用了以下第三方数据与模型，按各自许可使用。

## CHUBS 数据集

- 名称：CHUBS: A Large-Scale Dataset of Chu Bamboo Slip Script
- 来源：清华大学 NLP 组（THUNLP），COLING 2025
- 地址：https://huggingface.co/datasets/chen-yingfa/CHUBS
- 许可：Apache-2.0
- 本作品的使用方式：取其中 15 批材料的单字图，经筛选、缩放后用于训练与评测。`bundle_cnn/X_train.npy`、`X_val.npy`、`X_test.npy` 是由这些字图缩放得到的 64×64 灰度缓存，`index_*.json` 为其标签与出土出处。作品不随包分发 CHUBS 原始字图。

## Extract 的出处字段

`data/splits.csv` 中的 `batch`、`doc`、`slip` 三列直接来自 CHUBS 的文件名编码，用于在界面中显示形近字的「批次·篇·简号」。`path` 列为开发机上的原始路径，仅供溯源，运行时不读取。

## ResNet18 预训练权重

- 来源：torchvision 官方发布，ImageNet-1K 预训练
- 用途：仅用于设计方案 4.1、4.2 节的第一版对照实验；交付模型不含任何预训练权重

## 工具

- UI：Gradio（Apache-2.0）
- 训练：PyTorch（BSD-3-Clause）
- PDF 导出：LibreOffice（MPL-2.0）

## 引用

陈超、李赫孜、杨泽坤．基于深度集成学习的战国楚系简帛文字识别［J］．数字人文研究，2025，5(2)．

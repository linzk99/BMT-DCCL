# Boundary-aware and Competitive Contrastive Learning for Semi-supervised Medical Image Segmentation

This repository provides the official code for replicating experiments from the paper: **Boundary-aware and Competitive Contrastive Learning for Semi-supervised Medical Image Segmentation**

![]()
## Requirements
This repository is based on PyTorch 1.11.0, CUDA 11.7 and Python 3.8.18; All experiments in our paper were conducted on a single NVIDIA GeForce RTX 3090 GPU.

## Usage
1. Put the data in './BMT-DCCL/data';
2. Train the model;

```
# e.g., for 10% labels on ACDC
sh train.sh
```
3. Test the model;
```
# e.g., for 10% labels on ACDC
sh test.sh
```
## Acknowledgements:
Our code is adapted from [SS-Net](https://github.com/ycwu1997/SS-Net) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.


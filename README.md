# WPGST: Wavelet Pooling Group Swin Transformer For Superpixel Segmentation

Convolutional neural networks (CNNs) are widely used in superpixel segmentation. However, conventional CNNs mainly rely on local convolution and pooling operations, which limits their ability to capture global context and leads to feature aliasing. To address these limitations, we propose the wavelet pooling group Transformer (WPGT) for superpixel segmentation. First, a group Swin Transformer module is employed to capture both short- and long-range dependencies, while random convolution is applied to enhance inter-group interactions. Second, a wavelet pooling module is introduced to preserve essential feature information and mitigate feature degradation. Finally, a superpixel boundary-aware loss is designed to further improve spatial regularity. Experimental results on four datasets confirm that WPGT outperforms state-of-the-art approaches.

<img width="6715" height="4450" alt="Fig1" src="https://github.com/user-attachments/assets/123c6bf8-9fca-45dd-8d35-1f7b7a04e7e8" />
The overall architecture of WPGST comprises three key strategies: (1) group Swin Transformer with random convolution for efficiently capturing both local and global feature dependencies; (2) wavelet pooling for significantly preserving the main information; and (3) superpixel boundary-wise loss for stably enhancing grid regularity.


# ✨ Getting Start

# Environment Installation

Reference to Swin-Transformer (https://github.com/microsoft/Swin-Transformer) and SCN (https://github.com/fuy34/superpixel_fcn)

# Preparing Dataset
1. BSDS500: Following this link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
2. NYUDv2: Following this link: http://vcl.ucsd.edu/hed/nyu/
3. KITTI: Following this link: http://www.cvlibs.net/datasets/kitti/
4. DRIVE: Following this link: https://drive.grand-challenge.org/

Furthermore, Preprocessing of BSDS500 training data Following SCN (https://github.com/fuy34/superpixel_fcn)

# Training
    Run `python main.py` to start the program.

   ✨ It's worth mentioning that WPGST is trained exclusively on the BSDS500 training set and directly generates superpixels for NYUv2, KITTI, and DRIVE without requiring fine-tuning.

# Testing
1. Test BSDS500: Please run `run_infer_bsds.py`
2. Test NYUDv2: Please run `run_infer_nyu.py`
3. Test KITTI: Please run `run_infer_kitti.py`
4. Test DRIVE: Please run `run_infer_drive.py`

# Weights
We have placed the weight in the https://pan.baidu.com/s/1XLhxYGK5pVniRc-jxWqj6Q?pwd=217n password: 217n


# Acknowledgments

The basic code is partially from the below repos.
1. Swin-Transformer (https://github.com/microsoft/Swin-Transformer)
2. SCN (https://github.com/fuy34/superpixel_fcn)

# 📚 Cite Us 

✨ Please cite us if this work is helpful to you 

```bibtex
@ARTICLE{
  author={Xiaohong Jia, Yonghui Li, Xiaomei Guo, Yao Zhao, Guanghui Yan, and Zhengwen Huang},
  journal={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={WPGST: Wavelet Pooling Group Swin Transformer For Superpixel Segmentation}, 
  year={2026},
  volume={},
  pages={},
  keywords={Group Swin Transformer; Wavelet Pooling; Superpixel Segmentation},
  doi={}}

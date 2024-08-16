

# CorrAdaptor: Adaptive Local Context Learning for Correspondence Pruning (ECAI 2024)

[Wei Zhu](),[Yicheng Liu](),[Yuping He](),[Tangfei Liao](),[Xiaoqiu Xu](),
[Tao Wang](https://scholar.google.com/citations?user=TsDufoMAAAAJ&hl=en), 
[Tong Lu](https://cs.nju.edu.cn/lutong/index.htm)



[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2408.08134)


#### News
- **Aug 15, 2024:** Pre-trained models are released!
- **Aug 15, 2024:** Codes is released!

This repository contains the dataset, code and pre-trained models for our paper.


> **Abstract:** *In the fields of computer vision and robotics, accurate pixel-level correspondences are essential for enabling advanced tasks such as structure-from-motion and simultaneous localization and mapping. Recent correspondence pruning methods usually focus on learning local consistency through k-nearest neighbors, which makes it difficult to capture robust context for each correspondence. We propose CorrAdaptor, a novel architecture that introduces a dual-branch structure capable of adaptively adjusting local contexts through both explicit and implicit local graph learning. Specifically, the explicit branch uses KNN-based graphs tailored for initial neighborhood identification, while the implicit branch leverages a learnable matrix to softly assign neighbors and adaptively expand the local context scope, significantly enhancing the model's robustness and adaptability to complex image variations. Moreover, we design a motion injection module to integrate motion consistency into the network to suppress the impact of outliers and refine local context learning, resulting in substantial performance improvements. The experimental results on extensive correspondence-based tasks indicate that our CorrAdaptor achieves state-of-the-art performance both qualitatively and quantitatively.* 
<hr />


## Network Architecture
![](assets/corradaptor_framework.png)


## Get Started
### Dependencies and Installation

## Requirements

### Installation
We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below. 
```bash
conda create -n corradaptor python=3.8 --yes
conda activate corradaptor
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch --yes
python -m pip install -r requirements.txt

```

### Dataset
Follow the instructions provided [here](https://github.com/zjhthu/OANet) for downloading and preprocessing datasets. 
The packaged dataset should be put in the `data_dump/` and directory structure should be: 
```
$CorrAdaptor
    |----data_dump
      |----yfcc-sift-2000-train.hdf5
      |----yfcc-sift-2000-val.hdf5
      |----yfcc-sift-2000-test.hdf5
      |----sun3d-sift-2000-train.hdf5
      |----sun3d-sift-2000-val.hdf5
      |----sun3d-sift-2000-test.hdf5
      |----HPatches
        |----hpatches-sequences-release
        ...
```

## Training & Evaluation
1. If you have multiple gpus, it is recommended to use `train_multi_gpu.py` for training. 
```
# train by multiple gpus
CUDA_VISIBLE_DEVICES=0,1  python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=26331 --use_env train_multi_gpu.py

# train by single gpu
CUDA_VISIBLE_DEVICES=0  python train_single_gpu.py
```

2. Evaluation (Download pre-trained Models from links YFCC(https://pan.baidu.com/s/1t2CdZGwun_5WO1_c-iVflw?pwd=tib6) Sun3d(https://pan.baidu.com/s/1ERkhcLn_NmOcPgI5O_X9lA?pwd=4j7u))
```
python test.py
```

## Citation
If you use corradaptor, please consider citing:

    @inproceedings{zhu24corrAdaptor,
      title={CorrAdaptor: Adaptive Local Context Learning for Correspondence Pruning},
      author={Zhu, Wei and Liu, Yicheng and He, Yuping and Liao, Tangfei and Xu, Xiaoqiu and Wang, Tao and Lu, tong},
      booktitle={Proceedings of European Conference on Artificial Intelligence},
      year={2024}
    }

## Contact
If you have any questions, please contact taowangzj@gamil.com

**Acknowledgment:** This repo benefits from [OANet](https://github.com/zjhthu/OANet) and [CLNet](https://github.com/sailor-z/CLNet). Thanks for their wonderful works. 


---
<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=TaoWangzj/CorrAdaptor)

</details>


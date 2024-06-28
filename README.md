# CorrAdaptor [Paper]()

<!-- ## Description -->

In the fields of computer vision and robotics, accurate pixel-level correspondences are essential for enabling advanced tasks such as structure-from-motion and simultaneous localization and mapping. Recent correspondence pruning methods usually focus on learning local consistency through k-nearest neighbors, which makes it difficult to capture robust context for each correspondence. We propose CorrAdaptor, a novel architecture that introduces a dual-branch structure capable of adaptively adjusting local contexts through both explicit and implicit local graph learning. Specifically, the explicit branch uses KNN-based graphs tailored for initial neighborhood identification, while the implicit branch leverages a learnable matrix to softly assign neighbors and adaptively expand the local context scope, significantly enhancing the model’s robustness and adaptability to complex image variations. Moreover, we design a motion injection module to integrate motion consistency into the network to suppress the impact of outliers and refine local context learning, resulting in substantial performance improvements. The experimental results on extensive correspondence-based tasks indicate that our CorrAdaptor achieves state-of-the-art performance both qualitatively 1 and quantitatively. The code and pretrained models will be available.

<!-- ## Story figure -->

<p align="center">
    <img src="assets/corradaptor_story.png" width="70%"></a>
</p>


## 🏠 Overview

<!-- ## Architecture figure -->

<p align="center">
    <img src="assets/corradaptor_framework.png" width="0%"></a>
</p>


## 🛠️ Installation

* Clone this repo:

```
https://github.com/TaoWangzj/CorrAdaptor.git
cd CorrAdaptor
```

* Create a conda virtual environment and activate it:

```
conda create -n corradaptor python=3.9 -y
conda activate corradaptor
```

* Install requirements

```bash
pip install -r requirements.txt
```

## 🎯 Get Started

### Model Zoo

**YFCC100M**

| Model       | AUC@5° | link |
| ----------- | ------ | ---- |
| CorrAdaptor | 41.02  |      |

**Sun3D**

| Model       | AUC@5° | link |
| ----------- | ------ | ---- |
| CorrAdaptor | 9.33   |      |

### Evaluation

```bash
python test.py
```

### Training

**Train with single gpu**

```bash
CUDA_VISIBLE_DEVICES=0  python train_single_gpu.py
```

**Train with multi gpu**

```bash
CUDA_VISIBLE_DEVICES=0,1  python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=26331 --use_env train_multi_gpu.py
```

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).
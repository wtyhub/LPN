# [TCSVT] Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization 
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 


![](docs/index_files/visual.jpg#pic_center)


## LPN
[[Paper]](https://arxiv.org/abs/2008.11646) 

## NEWs
We upload the codes of [SAFA+ours](https://github.com/wtyhub/SAFA_LPN.git) and [CVFT+ours](https://github.com/wtyhub/CVFT_LPN.git)
## Prerequisites

- Python 3.6
- GPU Memory >= 8G
- Numpy > 1.12.1
- Pytorch 0.3+
- scipy == 1.2.1
- [Optional] apex (for float16) [Requirements](https://github.com/NVIDIA/apex#requirements) & [Quick Start](https://github.com/NVIDIA/apex#quick-start)

## Getting started
### Dataset & Preparation
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

Or download [CVUSA](http://cs.uky.edu/~jacobs/datasets/cvusa/) / [CVACT](https://github.com/Liumouliu/OriCNN). 

For CVUSA, I follow the training/test split in (https://github.com/Liumouliu/OriCNN). 

## Train & Evaluation
### Train & Evaluation University-1652
```  
sh run.sh
```
### Train & Evaluation CVUSA
```  
python prepare_cvusa.py  
sh run_cvusa.sh
```
### Train & Evaluation CVACT
```  
python prepare_cvact.py  
sh run_cvact.sh
```
## Citation

```bibtex
@ARTICLE{wang2021LPN,
  title={Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization}, 
  author={Wang, Tingyu and Zheng, Zhedong and Yan, Chenggang and Zhang, Jiyong and Sun, Yaoqi and Zheng, Bolun and Yang, Yi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2022},
  volume={32},
  number={2},
  pages={867-879},
  doi={10.1109/TCSVT.2021.3061265}}
```
```bibtex
@article{zheng2020university,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  journal={ACM Multimedia},
  year={2020}
}
```
## Related Work

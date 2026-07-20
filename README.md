# [TCSVT] Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization 
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 


![](docs/index_files/visual.jpg#pic_center)


## LPN
[[Paper]](https://arxiv.org/abs/2008.11646) [[中文解读]](https://zhuanlan.zhihu.com/p/365043015)


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
## 🔗 Ecosystem

<p align="center"><i>Explore our ecosystem for UAV & Spatial Intelligence 🚁 </i></p >

### 🚁 UAV & Spatial Intelligence

<p align="center"><b>🎓 The University-1652 Family</b></p >

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <a href=" ">
          <h3>🎓</h3>
          <b>University-1652</b>
        </a >
        <br><sub>Multi-view Multi-source Benchmark<br>Ground · Drone · Satellite · ACM MM'20</sub>
        <br><br>
        <a href="https://github.com/layumi/University1652-Baseline"><img src="https://img.shields.io/github/stars/layumi/University1652-Baseline.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/wtyhub/MuseNet">
          <h3>🌦️</h3>
          <b>University-WX</b>
        </a >
        <br><sub>Multi-Weather Extension on the Fly<br>Pattern Recognition'24</sub>
        <br><br>
        <a href="https://github.com/wtyhub/MuseNet"><img src="https://img.shields.io/github/stars/wtyhub/MuseNet.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/MultimodalGeo/GeoText-1652">
          <h3>💬</h3>
          <b>GeoText-1652</b>
        </a >
        <br><sub>Dense Text Extension<br>ECCV'24</sub>
        <br><br>
        <a href="https://github.com/MultimodalGeo/GeoText-1652"><img src="https://img.shields.io/github/stars/MultimodalGeo/GeoText-1652.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
    </tr>
  </table>
</div>

<p align="center"><b>🚀 New Open-Source Releases</b></p >

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <a href="https://github.com/YsongF/GeoFuse">
          <h3>🛰️</h3>
          <b>GeoFuse</b>
        </a >
        <br><sub>Road Maps as Free Geometric Priors </sub>
        <br><br>
        <a href="https://github.com/YsongF/GeoFuse"><img src="https://img.shields.io/github/stars/YsongF/GeoFuse.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/JT-Sun/UAVReason">
          <h3>🧠</h3>
          <b>UAVReason</b>
        </a >
        <br><sub>Aerial Scene Reasoning & Generation Benchmark</sub>
        <br><br>
        <a href="https://github.com/JT-Sun/UAVReason"><img src="https://img.shields.io/github/stars/JT-Sun/UAVReason.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/HaoDot/Video2BEV-Open">
          <h3>🗺️</h3>
          <b>Video2BEV</b>
        </a >
        <br><sub>Drone Video → Bird's-Eye-View</sub>
        <br><br>
        <a href="https://github.com/HaoDot/Video2BEV-Open"><img src="https://img.shields.io/github/stars/HaoDot/Video2BEV-Open.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/YaxuanLi-cn/PairUAV">
          <h3>🚁</h3>
          <b>PairUAV</b>
        </a >
        <br><sub>Paired UAV Data for Matching</sub>
        <br><br>
        <a href="https://github.com/YaxuanLi-cn/PairUAV"><img src="https://img.shields.io/github/stars/YaxuanLi-cn/PairUAV.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
    </tr>
  </table>
</div>

---

<p align="center">
  ⭐ If you find our projects helpful, a <b>star</b> is the best support! ⭐
</p >

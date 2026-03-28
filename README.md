# Semantic-MoSeg

> This is the offical implementation of the TITS paper: Semantic-MoSeg: Semantics-Assisted Moving-Obstacle Segmentation in Bird-Eye-View for Autonomous Driving. 

<div>
<img src="https://github.com/lab-sun/Semantic-MoSeg/blob/main/img/arc.png" alt="Semantic-MoSeg" width="720" height="480">
</div>

## 📚 Set up 
Our experiment is tested on Ubuntu 20.04 with Python 3.8.
- build environment
  ```
  conda create -n moseg python=3.8
  conda activate moseg
  pip install -r requirements.txt
  ```

## 📂 Dataset
We conduct the experiments based on NuScenes dataset and Lyft dataset.

## 🚀 Script and Visualization
```
  python3 train.py
```
Video: [https://www.youtube.com/GONw_DZgSmY](https://www.youtube.com/watch?v=GONw_DZgSmY)

## 🔗 Citation
```
@ARTICLE{meng2025semanticmoseg,
  author={Shiyu Meng and Yuxiang Sun},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Semantic-MoSeg: Semantics-Assisted Moving-Obstacle Segmentation in Bird-Eye-View for Autonomous Driving}, 
  year={2025},
  volume={26},
  number={7},
  pages={9251-9262},
  doi={10.1109/TITS.2025.3570058}}
```

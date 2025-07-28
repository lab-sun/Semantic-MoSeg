# Semantic-MoSeg: Semantics-Assisted Moving-Obstacle Segmentation in Bird-Eye-View for Autonomous Driving

This is the offical PyTorch implementation of  Semantics-Assisted Moving-Obstacle Segmentation in Bird-Eye-View for Autonomous Driving. 


<div>
<img src="https://github.com/lab-sun/Semantic-MoSeg/blob/main/img/arc.png" alt="描述文本" width="720" height="480">
</div>


Bird-eye-view (BEV) perception for autonomous driving has become popular in recent years. Among various BEV perception tasks, moving-obstacle segmentation is very important, since it can provide necessary information for downstream tasks, such as motion planning and decision making, in dynamic traffic environments. Many existing methods segment moving obstacles with LiDAR point clouds. The point-wise segmentation results can be easily projected into BEV since point clouds are 3-D data. However, these methods could not produce dense 2-D BEV segmentation maps, because LiDAR point clouds are usually sparse. Moreover, 3-D LiDARs are still expensive to vehicles. To provide a solution to these issues, this paper proposes a semantics-assisted moving-obstacle segmentation network using only low-cost visual cameras to produce segmentation results in dense 2-D BEV maps. Our network takes as input visual images from six surrounding cameras as well as the corresponding semantic segmentation maps at the current and previous moments, and directly outputs the BEV map for the current moment. We also propose a movable-obstacle segmentation auxiliary task to provide semantic information to further benefit moving-obstacle segmentation.


## 🔑 Set up 
Our experiment is tested on Ubuntu 20.04 with Python 3.8.
- build environment
  ```
  conda create -n moseg python=3.8
  conda activate moseg
  pip install -r requirements.txt
  ```

## 📚 DataSet
We conduct the experiments based on NuScenes dataset and Lyft dataset.


## 💡  Script and Visualization

```
  python3 train.py
  ```

Video: [https://www.youtube.com/GONw_DZgSmY](https://www.youtube.com/watch?v=GONw_DZgSmY)

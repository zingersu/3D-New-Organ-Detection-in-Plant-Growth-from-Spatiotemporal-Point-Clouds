3D New Organ Detection in Plant Growth from Spatiotemporal Point Clouds
=====
Prerequisites
------
The code only has a tensorflow version now, and its corresponding configurations are as follows:<br>
* All deep networks run under Ubuntu 18.04<br>
* Tensorflow version:<br>
    * Python == 3.16.3<br>
    * Tensorflow == 1.13.1<br>
    * CUDA == 10.0<br>

Introduction
------
Automatic plant growth monitoring is an important task in modern agriculture for maintaining high crop yield and boosting the breeding procedure. The advancement of 3D sensing technology has made 3D point clouds to be a better data form on presenting plant growth than images, as the new organs are easier identified in 3D space and the occluded organs in 2D can also be conveniently separated in 3D. Despite the attractive characteristics, analysis on 3D data can be quite challenging. <br>
<br>
3D-NOD is a framework to detect new organs from time-series 3D plant data by spatiotemporal point cloud deep semantic segmentation. The design of 3D-NOD framework drew inspiration from how a well-experienced human utilizes spatiotemporal information to identify growing buds from a plant at two different growth stages. The framework by introducing the Backward & Forward Labeling, the Registration & Mix-up, and the Humanoid Data Augmentation step, make our backbone network be trained to recognize growth events with organ correlation from both temporal and spatial domains. Our framework has shown better sensitivity at segmenting new organs against the conventional way of using a network to conduct direct semantic segmentation.<br>
![Fig  1](https://github.com/user-attachments/assets/ac0ad520-3351-48d9-9688-ccb081cf455a)

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
<p align="center">
  <strong><em>A brief comparison of the conventional new organ detection framework and our spatiotemporal framework for new organ detection. (a) is the conventional way and (b) is ours.</em></strong>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ac0ad520-3351-48d9-9688-ccb081cf455a" alt="Fig 1" width="70%"/>
</p>
<br>
<p align="center">
  <strong><em>Our detailed 3D-NOD framework for growth event detection. (a) is the training pipeline and (b) is the testing pipeline. The green points represent old organ, and purple points for new organ. </em></strong>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/8ba3d119-a8c9-422a-8e7f-18632ca3ed36" alt="Fig 2" width="100%"/>
</p>

Quick Start
------
This project contains three folders<br>
folder <strong>[data_preprocessing]</strong> contains all the code to process the raw dataset and converts the processed data into .h5 format for network training and testing<br>
folder <strong>[backbone_network]</strong> contains the dgcnn model that serves as the main architecture for 3D-NOD, it also contains parts of the raw dataset and processed .h5 files which can be used to train and test the model<br>
folder <strong>[data_post-processing]</strong> contains all the code for the Split & Refinement phase, which post-processes the test results to obtain the final quantitative results.<br>

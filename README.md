3D-NOD
------
This repo contains the official data and code for our paper:

3D-NOD: 3D New Organ Detection in Plant Growth by a Spatiotemporal Point Cloud Deep Segmentation Framework.<br>
[D. Li†](https://davidleepp.github.io/), F. Ahmed†, and Z. Wang†<br>
† Equal contribution<br>
Published online on *Plant Phenomics* in 2025<br>
[[Paper](https://www.sciencedirect.com/science/article/pii/S2643651525000081)]
[[11-minute video presentation](https://www.bilibili.com/video/BV1HGktYoEwP/)]


Prerequisites
------
The code only has a TensorFlow version now, and its corresponding configurations are as follows:<br>
* All deep networks run under Ubuntu 18.04<br>
* Tensorflow version:<br>
    * Python == 3.6.13<br>
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
This project contains three folders.<br>
folder <strong>[data_preprocessing]</strong> contains all the code to preprocess the raw dataset and convert the processed data into .h5 format for network training and testing.<br>
folder <strong>[backbone_network]</strong> contains the [DGCNN](https://arxiv.org/abs/1801.07829) model that serves as the main architecture for 3D-NOD, it also contains parts of the raw dataset and processed .h5 files that can be used to train and test the model.<br>
folder <strong>[data_post-processing]</strong> contains all the code for the Split & Refinement steps in the testing phase, which acts as the postprocessing on the predicted results from DGCNN for final quantitative and qualitative results.<br>
<br>

<strong><em>Data_preprocessing</em></strong><br>
Raw data needs to be preprocessed before being fed into the network for training or testing, and preprocessing of raw data can be achieved with the following code.<br>
* file <strong>[00pcd_to_txt.py]</strong> is used to convert the PCD files into TXT files for subsequent processing.<br>
* file <strong>[01norm.py]</strong> is used to normalize the original TXT files in 3D space for subsequent ICP registration.<br>
* file <strong>[02FPS_once.py]</strong> is used to downsample the file to 2048 points per point cloud using FPS.<br>
* file <strong>[03ICP.py]</strong> is used to align the point clouds of every two adjacent moments and <strong>use the T+1 moment point cloud and the T moment point cloud for merge</strong>.<br>
* file <strong>[04add_index_for_Reg_folder.py]</strong> is used to add a time index to each of the point in the merged point cloud (0 for the latest moment, and 1 for the previous moment), which is then fed into the network as a supervisory signal, allowing the network to have the ability to "distinguish" the two point clouds from two different times in the merged point cloud.<br>
* file <strong>[05dis_train_from_test.py]</strong> is used to divide the point clouds into a training set and a testing set. The files containing "A" and "B" in their names are used as training sets, and files containing "C" in their names are used as test sets.<br>
* file <strong>[06Aug_for_train.py]</strong> is used to augment (default 10x) the training set with Humanoid Data Augmentation (HDA).<br>
* file <strong>[07script.py]</strong> and file <strong>[08Convert_txt_to_H5_file.py]</strong> are used together to generate the .h5 format file for network input.<br>
<br>

<strong><em>Backbone_network</em></strong><br>
The folder contains all code for training DGCNN network in the TensorFlow environment. The previously generated .h5 file is passed on to the network as the input.<br>
* folder <strong>[data]</strong> contains <strong>part of the training set</strong> and all of the testing set, and their corresponding .h5 files, which can be used directly to train the model.<br>
* folder <strong>[models]</strong> contains the semantic segmentation and <strong>instance segmentation network</strong> of DGCNN, here we use <strong>"pointnet2_part_seg.py"</strong> to implement the task of semantic segmentation of old and new organs, the code defines the network structure as well as the loss function.<br>
* folder <strong>[part_seg]</strong> contains the code for DGCNN's entire training and testing processes.<br>
   * file <strong>[00train.py]</strong> is used to train the model using the training set.<br>
   * file <strong>[01evaluate.py]</strong> is used to do testing (predictions) on best trained model parameters.<br>
   * file <strong>[02eval_iou_accuracy.py]</strong> is used to compute quantitative metrics for the task of semantic segmentation of old and new organs. But note that the calculated metrics are not the final metrics since the input is a merged point cloud that has to be further separated and refined.<br>

<strong>Note:</strong> When downloading files from this repository, due to github's limitations, files larger than 50 MB need to be downloaded separately, otherwise you will get an error file that cannot open.<br>
<br>

<strong><em>Data_post-processing</em></strong><br>
Since the DGCNN network takes the aligned point cloud as input and maintains spatial correspondence in its output characteristics. Therefore the raw output of the network does not directly reflect the appearance of new organs in the plant sequence, and further processing of the output of the DGCNN network is required to obtain the new organ detection results for each plant in the sequence.<br>
* file <strong>[00from_txt_to_folder.py]</strong> is used to convert the two TXT files output from the network into two folders, which contain one-to-one correspondence of the point cloud data to facilitate subsequent processing.<br>
* file <strong>[01Splitment & Refinement.py]</strong> is used to split the aligned point cloud into two plant point clouds at adjacent moments, and subsequently process the point clouds belonging to the same moment using the Refinement method in this paper.<br>
* file <strong>[02eval_iou_accuracy.py]</strong> is used to calculate quantitative indicators for all plants in the test set.

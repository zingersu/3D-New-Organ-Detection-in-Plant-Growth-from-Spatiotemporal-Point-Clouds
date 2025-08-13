3D New Organ Detection in Plant Growth from Spatiotemporal Point Clouds
=====
Prerequisites
------
The code only has a tensorflow version now, and its corresponding configurations are as follows:<br>
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
folder <strong>[data_preprocessing]</strong> contains all the code to process the raw dataset and converts the processed data into .h5 format for network training and testing<br>
folder <strong>[backbone_network]</strong> contains the dgcnn model that serves as the main architecture for 3D-NOD, it also contains parts of the raw dataset and processed .h5 files which can be used to train and test the model<br>
folder <strong>[data_post-processing]</strong> contains all the code for the Split & Refinement phase, which post-processes the test results to obtain the final quantitative and qualitative results.<br>
<br>

<strong><em>data_preprocessing</em></strong><br>
Raw data needs to be preprocessed before it can be fed into networks for training or testing, and pre-processing of raw data can be achieved with the following code.<br>
* file <strong>[00pcd_to_txt.py]</strong> is used to convert the PCD files into TXT files for subsequent processing.<br>
* file <strong>[01norm.py]</strong> is used to normalise the original TXT files for subsequent ICP registration.<br>
* file <strong>[02FPS_once.py]</strong> is used to downsample the points in the file to 2048 points using FPS.<br>
* file <strong>[03ICP.py]</strong> is used to match the point clouds of neighbouring moments two by two and use the T+1 moment point cloud and the T moment point cloud for alignment.<br>
* file <strong>[04add_index_for_Reg_folder.py]</strong> is used to add a time index to the aligned point cloud, which is fed into the network as a supervisory signal, allowing the network to compare point clouds at different moments in time.<br>
* file <strong>[05dis_train_from_test.py]</strong> is used to divide the point clouds into a training set and a testing set. The files containing "A" and "B" in their names are used as training sets, and files containing "C" in their names are used as test sets.<br>
* file <strong>[06Aug_for_train.py]</strong> is used to augment (default 10x) the training set with data using humanoid methods.<br>
* file <strong>[07script.py]</strong> and file <strong>[08Convert_txt_to_H5_file.py]</strong> are used together to generate the .h5 format file for network input.<br>
<br>

<strong><em>backbone_network</em></strong><br>
The folder contains all code for training DGCNN network in tensorFlow environment. After getting the .h5 file, pass it as input to the network.<br>
* folder <strong>[data]</strong> contains part of the training set and all of the test set, and their corresponding .h5 files, which can be used directly to train the model.<br>
* folder <strong>[models]</strong> contains the semantic segmentation and instance segmentation network of DGCNN, here we use <strong>”pointnet2_part_seg.py“</strong> to implement the task of semantic segmentation of old and new organs, the code contains the network structure and loss function.<br>
* folder <strong>[part_seg]</strong> contains the code for DGCNN's entire training and testing processes.<br>
   * file <strong>[00train.py]</strong> is used to train the model parameters using the training set.<br>
   * file <strong>[01evaluate.py]</strong> is used to test on a test set using the model parameters of the best saved model to obtain predictions.<br>
   * file <strong>[02eval_iou_accuracy.py]</strong> is used to compute quantitative metrics for the task of semantic segmentation of old and new organs. But note that the calculated metrics are not the final metrics since the input is the aligned point cloud.<br>

<strong>Note:</strong> When downloading files from this repository, due to github's limitations, files larger than 50 megabytes need to be downloaded separately, otherwise you will get an error file that cannot be opened.<br>
<br>

<strong><em>data_post-processing</em></strong><br>
Since the DGCNN network takes the aligned point cloud as input and maintains spatial correspondence in its output characteristics. Therefore the raw output of the network does not directly reflect the appearance of new organs in the plant sequence, and further processing of the output of the DGCNN network is required to obtain the new organ detection results for each plant in the sequence.<br>
* file <strong>[00from_txt_to_folder.py]</strong> is used to convert the two TXT files output from the network into two folders, which contain one-to-one correspondence of the point cloud data to facilitate subsequent processing.<br>
* file <strong>[01Splitment & Refinement.py]</strong> is used to split the aligned point cloud into two plant point clouds at adjacent moments, and subsequently process the point clouds belonging to the same moment using the Refinement method in this paper.<br>
* file <strong>[02eval_iou_accuracy.py]</strong> is used to calculate quantitative indicators for all plants in the test set.

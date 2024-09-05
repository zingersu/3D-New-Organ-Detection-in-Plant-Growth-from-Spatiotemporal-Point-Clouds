import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:, 0:3]
    ins_label = (data[:, 3]).astype(int)
    find_index = np.where(ins_label >= 1)
    time_label = (data[:, -1]).astype(int)

    return point_xyz, ins_label, time_label


def change_scale(data):
    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    data[:, 0:3] = data[:, 0:3] - xyz_move
    # scale
    scale = np.max(data[:, 0:3])

    return data[:, 0:3] / scale



def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


'''
    input:  TXT dataset in the form of XYZLabel
            Txt file for training sample names
            Txt file for testing sample names
	output: h5 File for training or testing
'''



if __name__ == "__main__":
    root = os.path.join(ROOT_DIR, "backbone_network\\data")

    #  Reading training or testing data
    plant_classes = {}      # 空字典
    plant_classes['train'] = [line.rstrip() for line in open(os.path.join(root, 'train_ids.txt'))]
    datapath = [(os.path.join(root, "Aug_train_data", plant_classes['train'][i])) for i in range(len(plant_classes['train']))]
    print('The size of %s data is %d' % ("dataset", len(datapath)))

    save_path = root + "\\" + "TRAIN.h5"  # save path of h5 File

    num_sample = 4096
    files_count = len(datapath)
    DATA_ALL = []

    for index in tqdm(range(len(datapath)), total=len(datapath)):
        fn = datapath[index]

        current_data, current_ins_label, current_time_label = loadDataFile(fn)

        for i in range(len(current_ins_label)):
            basename = os.path.basename(fn)

            if current_ins_label[i] == 1:     # A label of 1 indicates that the point belongs to an old organ of the plant, and the semantic label assigned to the point is 0
                current_ins_label[i] = 0
            elif current_ins_label[i] == 2:   # A label of 2 indicates that the point belongs to a new organ of the plant, and the semantic label assigned to the point is 1
                current_ins_label[i] = 1
            else:
                current_ins_label[i] = 0    # consider all noise as points in the old organ

        change_data = change_scale(current_data)

        data_label = np.column_stack((change_data, current_ins_label, current_time_label))
        DATA_ALL.append(data_label)
    print(np.asarray(DATA_ALL).shape)
    output = np.vstack(DATA_ALL)
    output = output.reshape(files_count, num_sample, 5)

    print(output.shape)

    if not os.path.exists(save_path):
        with h5py.File(save_path, 'w') as f:

            f['data'] = output[:, :, 0:3]
            f['organ_seg'] = output[:, :, 3]
            f['time_seg'] = output[:, :, 4]

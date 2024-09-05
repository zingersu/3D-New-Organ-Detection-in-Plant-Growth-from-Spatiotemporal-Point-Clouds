import numpy as np
import os
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(file_name.split("_")[1])
    return Initial

ini_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\train_data")
save_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\Aug_train_data")
all_files = sorted(os.listdir(ini_folder), key=sort_by_number)

if os.path.exists(save_folder) == False:
    os.makedirs(save_folder)

rotate_frequency = 10
batch_num = 1
for k in range(0, len(all_files), batch_num):
    current_files = all_files[k : k + batch_num]
    current_files = current_files * 10
    new_files = []
    for j in range(len(current_files)):
        current_file_name = current_files[j]
        new_name = os.path.splitext(current_file_name)[0] + "_" + str(j) + ".txt"
        new_files.append(new_name)

    for file_name, new_file_name, i in zip(current_files, new_files, range(rotate_frequency)):
        full_path = os.path.join(ini_folder, file_name)
        data = np.loadtxt(full_path)
        data_coo = data[:, :3]
        data_all_label = data[:, 3:]
        data_time_label = data[:, -1]

        data_T_time_index = np.where(data_time_label==0)
        data_T_time_coo = data_coo[data_T_time_index]
        M = len(data_T_time_coo)

        data_T_plus_time_index = np.where(data_time_label==1)
        data_T_plus_time_coo = data_coo[data_T_plus_time_index]
        N = len(data_T_plus_time_coo)

        arr_T = np.ones(M).reshape(-1,1)
        homo_T_coo = np.hstack((data_T_time_coo, arr_T))
        arr_T_plus = np.ones(N).reshape(-1,1)
        homo_T_plus_coo = np.hstack((data_T_plus_time_coo, arr_T_plus))

        rotate_angles = 2 * np.pi / rotate_frequency



        # Processing the point cloud at T time
        each_rotate_angles = i * rotate_angles
        T_rotation_matrix = np.array([[np.cos(each_rotate_angles), 0, -np.sin(each_rotate_angles)],
                                    [0, 1, 0],
                                    [np.sin(each_rotate_angles), 0, np.cos(each_rotate_angles)]])
        T_translation_vector = np.zeros((3,1))

        T_homogeneous_matrix = np.eye(4)
        T_homogeneous_matrix[:3, :3] = T_rotation_matrix
        T_homogeneous_matrix[:3, 3:4] = T_translation_vector

        B = np.dot(T_homogeneous_matrix, homo_T_coo.T).T



        # Processing the point cloud at T+1 time
        random_agitation = random.randint(0,5) * np.pi / 180
        T_plus_rotation_matrix = np.array([[np.cos(each_rotate_angles + random_agitation), 0, -np.sin(each_rotate_angles + random_agitation)],
                                    [0, 1, 0],
                                    [np.sin(each_rotate_angles + random_agitation), 0, np.cos(each_rotate_angles + random_agitation)]])

        mean = 0
        std_dev = 0.05
        array = np.random.normal(mean, std_dev, (3, 1))
        clipped_array = np.clip(array, -0.15, 0.15)
        T_plus_translation_vector = clipped_array

        T_plus_homogeneous_matrix = np.eye(4)
        T_plus_homogeneous_matrix[:3, :3] = T_plus_rotation_matrix
        T_plus_homogeneous_matrix[:3, 3:4] = T_plus_translation_vector

        C = np.dot(T_plus_homogeneous_matrix, homo_T_plus_coo.T).T


        new_coo = np.vstack((B, C))
        new_data = np.hstack((new_coo[:, :3], data_all_label))
        save_path = os.path.join(save_folder, new_file_name)
        np.savetxt(save_path, new_data, delimiter=" ", fmt="%f %f %f %d %d %d")

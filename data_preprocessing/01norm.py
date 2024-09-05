import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def normalize_coo(mx):
    data = np.genfromtxt(mx, delimiter=" ")
    new_data = data[:, :3]
    label = data[:, 3:]
    center = np.mean(new_data, axis=0)
    normalized_points = new_data - center
    max_distance = np.max(np.linalg.norm(normalized_points, axis=1))
    normalized_points /= max_distance
    data_xyz = np.array(normalized_points)
    final_data = np.hstack((data_xyz,label))
    return final_data


def sort_by_number(file_name):
    Initial = int(file_name.split('_')[0])
    return Initial


all_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\Data_txt")
all_folder_norm = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_txt")
all_files = sorted(os.listdir(all_folder), key=sort_by_number)

if os.path.exists(all_folder_norm)==False:
    os.makedirs(all_folder_norm)


for file in all_files:
    file_path = os.path.join(all_folder, file)
    A = normalize_coo(file_path)
    np.savetxt(all_folder_norm + "\\" + "norm_" + file, A, delimiter=" ", fmt='%f %f %f %d %d')


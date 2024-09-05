import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(file_name.split('_')[1])
    return Initial


ini_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps")
reg_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps_ICP")
save_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps_ICP_index")

if os.path.exists(save_folder) == False:
    os.makedirs(save_folder)

ini_all_files = sorted(os.listdir(ini_folder),key=sort_by_number)
reg_all_files = sorted(os.listdir(reg_folder),key=sort_by_number)
reg_all_files = list(reg_all_files)
ini_file_numbers = len(ini_all_files)

for i in range(ini_file_numbers - 1):
    source_file_norm = ini_all_files[i]
    target_file_norm = ini_all_files[i + 1]

    parts_to_check = [source_file_norm.split('_')[2] == target_file_norm.split('_')[2],
                      source_file_norm.split('_')[3] == target_file_norm.split('_')[3],
                      source_file_norm.split('_')[4] == target_file_norm.split('_')[4]]

    if all(parts_to_check):
        continue
    else:
        reg_all_files.insert(i," ")

reg_all_files.append(" ")
reg_all_files = np.array(reg_all_files)


for j,k in zip(range(ini_file_numbers-1),range(ini_file_numbers-1)):
    source_file_norm = ini_all_files[j]
    target_file_norm = ini_all_files[j + 1]

    source_path = os.path.join(ini_folder, source_file_norm)
    target_path = os.path.join(ini_folder, target_file_norm)

    source_data = np.loadtxt(source_path)
    M = source_data.shape[0]
    t_time_label = np.full(M,0).reshape(-1,1)

    target_data = np.loadtxt(target_path)
    N = target_data.shape[0]
    t_plus_one_time_label = np.full(N,1).reshape(-1,1)


    reg_file = reg_all_files[k]
    if reg_file != " ":
        reg_path = os.path.join(reg_folder,reg_file)
        reg_data = np.loadtxt(reg_path)
        t_time_data = reg_data[:M, :]
        t_time_data = np.hstack((t_time_data,t_time_label))

        t_plus_one_time_data = reg_data[M:, :]
        t_plus_one_time_data = np.hstack((t_plus_one_time_data,t_plus_one_time_label))

        final_reg_data = np.vstack((t_time_data,t_plus_one_time_data))
        save_path = os.path.join(save_folder, reg_file)
        np.savetxt(save_path, final_reg_data, delimiter=" ", fmt="%f %f %f %d %d %d")
    else:
        continue
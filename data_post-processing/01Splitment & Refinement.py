import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(os.path.splitext(file_name)[0].split("_")[1])
    return Initial

base_path = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_C_fps_2048")       # Pre-alignment, post-downsampling test set files
ICP_ini_gt_path = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_C_fps_ICP_index")        # Downsampled and aligned test set files
ICP_pred_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\out_pred")           # Predicted folder output by 00from_txt_to_folder.py
 # The files in "ICP_ini_gt_path" and "ICP_pred_path" correspond to each other.

save_folder_T = os.path.join(ROOT_DIR, "Epoch46\\test_results\\pred_2048_T")        # Store the files at moment T after Splitment
save_folder_T1 = os.path.join(ROOT_DIR, "Epoch46\\test_results\\pred_2048_T+1")     # Store the files at moment T+1 after Splitment
final_sem_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\pred_2048_refinement")      # Store the files after Splitment & Refinement

if os.path.exists(save_folder_T)==False:
    os.makedirs(save_folder_T)
if os.path.exists(save_folder_T1)==False:
    os.makedirs(save_folder_T1)
if os.path.exists(final_sem_path)==False:
    os.makedirs(final_sem_path)

all_ICP_files = sorted(os.listdir(ICP_ini_gt_path), key=sort_by_number)
all_pred_ICP_files = sorted(os.listdir(ICP_pred_path), key=sort_by_number)
check_same_sequence = []
for file_name in all_ICP_files:
    Sequence_judge = "_".join([file_name.split("_")[2], file_name.split("_")[3], file_name.split("_")[4]])
    check_same_sequence.append(Sequence_judge)

sequence_file_num = []     # Number of post-alignment files stored for each plant sequence, and corresponds exactly to the sequence name in "sequence_file_name"
sequence_file_name = []
for sequence_file in check_same_sequence:
    if sequence_file not in sequence_file_name:
        sequence_file_name.append(sequence_file)
    else:
        continue

for sequence_judge in sequence_file_name:
    file_num = check_same_sequence.count(sequence_judge)
    sequence_file_num.append(file_num)

# So far, we can obtain the number of files for each sequence, as well as the names of the different sequences; and the elements in them are corresponding


start_idx = 0
end_idx = 0
for i in range(len(sequence_file_num)):
    end_idx = start_idx + sequence_file_num[i]
    current_files = all_pred_ICP_files[start_idx:end_idx]
    gt_current_files = all_ICP_files[start_idx: end_idx]


    for j, real_file_name in zip(range(len(current_files)), gt_current_files):
        pred_full_path = os.path.join(ICP_pred_path, current_files[j])
        pred_ICP_data = np.loadtxt(pred_full_path)

        pred_T_data = pred_ICP_data[:2048, :]
        pred_T_plus_data = pred_ICP_data[2048:, :]

        pred_T_name = real_file_name.split(" & ")[0] + ".txt"
        pred_T_plus_name = "_".join(real_file_name.split("_")[:-1]) + "_" + real_file_name.split(" & ")[1]

        np.savetxt(os.path.join(save_folder_T, pred_T_name), pred_T_data, delimiter=" ", fmt="%f %f %f %d %d")
        np.savetxt(os.path.join(save_folder_T1, pred_T_plus_name), pred_T_plus_data, delimiter=" ", fmt="%f %f %f %d %d")

    start_idx = end_idx



all_files = sorted(os.listdir(base_path), key=sort_by_number)       # Use the raw coordinate information of the files in this folder
all_T_files = sorted(os.listdir(save_folder_T), key=sort_by_number)
all_T_plus_files = sorted(os.listdir(save_folder_T1), key=sort_by_number)       # Use the files in these two folders for refinement.


restart_idx = 0
end_again_idx = 0
another_start_idx = 0
another_end_idx = 0
for k in range(len(sequence_file_num)):
    end_again_idx = restart_idx + sequence_file_num[k]
    T_current_files = all_T_files[restart_idx:end_again_idx]
    T_plus_current_files = all_T_plus_files[restart_idx: end_again_idx]

    T_current_files.append(" ")
    T_plus_current_files.insert(0," ")

    each_sequence_file_num = len(T_current_files)
    another_end_idx = another_start_idx + each_sequence_file_num
    ini_real_file_name = all_files[another_start_idx:another_end_idx]

    for m, n, ini_file_name in zip(T_current_files, T_plus_current_files, ini_real_file_name):
        if n == " ":
            first_file_path = os.path.join(save_folder_T, m)
            first_file_data = np.loadtxt(first_file_path)
            first_file_pred_label = first_file_data[:, 4:]

            first_gt_path = os.path.join(base_path, ini_file_name)
            first_gt_data = np.loadtxt(first_gt_path)
            first_gt_coo = first_gt_data[:, :3]
            first_pred_file_data = np.hstack((first_gt_coo, first_file_pred_label))
            save_path = os.path.join(final_sem_path, ini_file_name)
            np.savetxt(save_path, first_pred_file_data, delimiter=" ", fmt="%f %f %f %d")

        elif m != " " and n != " ":
            ini_file_path = os.path.join(save_folder_T, m)
            ini_file_data = np.loadtxt(ini_file_path)
            ini_file_label = ini_file_data[:, 4:]

            ICP_file_path = os.path.join(save_folder_T1, n)
            ICP_file_data = np.loadtxt(ICP_file_path)
            ICP_file_label = ICP_file_data[:, 4:]

            final_sem_label = np.where((ini_file_label == 0) & (ICP_file_label == 0), 0,1)

            gt_path = os.path.join(base_path, ini_file_name)
            gt_data = np.loadtxt(gt_path)
            gt_coo = gt_data[:, :3]

            refinement_data = np.hstack((gt_coo, final_sem_label))
            save_path = os.path.join(final_sem_path, ini_file_name)
            np.savetxt(save_path, refinement_data, delimiter=" ", fmt="%f %f %f %d")
        elif m == " ":
            final_file_path = os.path.join(save_folder_T1, n)
            final_file_data = np.loadtxt(final_file_path)
            final_file_pred_label = final_file_data[:, 4:]

            final_gt_path = os.path.join(base_path, ini_file_name)
            final_gt_data = np.loadtxt(final_gt_path)
            final_gt_coo = final_gt_data[:, :3]
            final_pred_file_data = np.hstack((final_gt_coo, final_file_pred_label))
            save_path = os.path.join(final_sem_path, ini_file_name)
            np.savetxt(save_path, final_pred_file_data, delimiter=" ", fmt="%f %f %f %d")

    restart_idx = end_again_idx
    another_start_idx = another_end_idx

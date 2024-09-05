import os
import shutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(file_name.split("_")[1])
    return Initial

ini_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps_ICP_index")
save_train = os.path.join(ROOT_DIR, "backbone_network\\data\\train_data")
save_test = os.path.join(ROOT_DIR, "backbone_network\\data\\test_data")
all_files = sorted(os.listdir(ini_folder), key=sort_by_number)

if os.path.exists(save_train) == False:
    os.makedirs(save_train)

if os.path.exists(save_test) == False:
    os.makedirs(save_test)

sequence_name = ["A","B"]
for file_name in all_files:
    if file_name.split("_")[4] in sequence_name:
        full_path = os.path.join(ini_folder,file_name)
        shutil.copy(full_path,save_train)
    else:
        full_path = os.path.join(ini_folder, file_name)
        shutil.copy(full_path, save_test)
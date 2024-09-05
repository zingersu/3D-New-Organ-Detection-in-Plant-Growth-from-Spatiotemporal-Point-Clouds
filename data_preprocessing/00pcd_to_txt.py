import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

pcd_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\Data_V3")
txt_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\Data_txt")

if os.path.exists(txt_folder)==False:
    os.makedirs(txt_folder)

all_files = sorted(os.listdir(pcd_folder))
file_number = len(all_files)

for i in range(file_number):
    pcd_file = all_files[i]
    if pcd_file.split("_")[-1] == "V3.pcd":
        output_file = pcd_file.replace("_centre_filter_NOS_V3.pcd", ".txt")
        output_file = f"{i+1}_{output_file}"
    else:
        output_file = pcd_file.replace("_centre_filter_NOS_V2.pcd", ".txt")
        output_file = f"{i+1}_{output_file}"

    pcd_file_path = pcd_folder + "\\" + pcd_file
    output_file_path = txt_folder + "\\" + output_file

    with open(pcd_file_path, 'r') as file:
        lines = file.readlines()

    data_start_index = lines.index("DATA ascii\n") + 3
    data_lines = lines[data_start_index:]
    data_lines = [line.strip() for line in data_lines]

    with open(output_file_path, 'w') as output_file:
        for line in data_lines:
            output_file.write(line + "\n")
    print("数据已写入到", output_file_path)
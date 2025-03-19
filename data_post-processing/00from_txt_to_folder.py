import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

pred_base_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\out_pred")
pre_file_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\out_pred.txt")     # Path to "out_pred.txt" for model output
gt_base_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\out_gt")
gt_file_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\out_gt.txt")    # Path to "out_gt.txt" for model output
pre_data = np.loadtxt(pre_file_path)
gt_data = np.loadtxt(gt_file_path)
pointsize = 4096
file_num = int(len(pre_data) / pointsize)

if os.path.exists(pred_base_path)==False:
    os.makedirs(pred_base_path)
if os.path.exists(gt_base_path)==False:
    os.makedirs(gt_base_path)

pred_input_file_path = pred_base_path + "\\" + 'pred'
gt_input_file_path = gt_base_path + "\\" + 'gt'

for i in range(file_num):
    pred_temp_path= pred_input_file_path + "_" + str(i)+'.txt'
    pred_data = pre_data[i*4096:(i+1)*4096, :]
    np.savetxt(pred_temp_path, pred_data, delimiter=" ", fmt="%f %f %f %d %d")

    gt_temp_path = gt_input_file_path + "_" + str(i)+'.txt'
    real_data = gt_data[i*4096:(i+1)*4096, :]
    np.savetxt(gt_temp_path, real_data, delimiter=" ", fmt="%f %f %f %d %d")
import os
import numpy as np

def sort_by_number(file_name):
    Initial = int(os.path.splitext(file_name)[0].split("_")[1])
    return Initial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

GT_path = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_C_fps_2048")
PRED_path = os.path.join(ROOT_DIR, "Epoch46\\test_results\\pred_2048_refinement")
all_files = sorted(os.listdir(GT_path), key=sort_by_number)
plant_number = len(all_files)
NUM_CLASSES = 2

# plant_species = ["benthi", "m82D", "sorghum"]
# idx_plant_files = []
# for file in all_files:
#     if file.split('_')[2] == plant_species[1]:
#         idx_plant_files.append(file)
# plant_number = len(idx_plant_files)         # Quantitative indicators for the calculation of different plant species


# Initialize...
# acc and macc
total_true = 0
total_seen = 0
true_positive_classes = np.zeros(NUM_CLASSES)
true_negative_classes = np.zeros(NUM_CLASSES)
false_positive_classes = np.zeros(NUM_CLASSES)
false_negative_classes = np.zeros(NUM_CLASSES)
positive_classes = np.zeros(NUM_CLASSES)
gt_classes = np.zeros(NUM_CLASSES)
# mIoU
ious = np.zeros(NUM_CLASSES)
totalnums = np.zeros(NUM_CLASSES)


for i in range(plant_number):
    pred_data = np.loadtxt(os.path.join(PRED_path, all_files[i]))
    # pred_data = np.loadtxt(os.path.join(PRED_path, idx_plant_files[i]))       # Quantitative indicators for the calculation of different plant species
    pred_sem = pred_data[:, -1].reshape(-1).astype(int)
    gt_data = np.loadtxt(os.path.join(GT_path, all_files[i]))
    # gt_data = np.loadtxt(os.path.join(GT_path, idx_plant_files[i]))
    gt_sem = gt_data[:, -2].reshape(-1).astype(int)

    for k in range(len(gt_sem)):
        if gt_sem[k]==1:
            gt_sem[k] = 0
        elif gt_sem[k]==2:
            gt_sem[k] = 1
        else:
            gt_sem[k] = 0

    for j in range(gt_sem.shape[0]):
        gt_l = int(gt_sem[j])
        pred_l = int(pred_sem[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[pred_l] += int(gt_l==pred_l)
        false_positive_classes[pred_l] += int(gt_l!=pred_l)
        false_negative_classes[gt_l] += int(gt_l!=pred_l)

precision = np.zeros(NUM_CLASSES)
recall = np.zeros(NUM_CLASSES)

LOG_FOUT = open(os.path.join(ROOT_DIR, "Epoch46\\test_results\\Epoch46.txt"), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# semantic results
iou_list = []
for i in range(NUM_CLASSES):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    iou_list.append(iou)

precision = true_positive_classes / (false_positive_classes+true_positive_classes)
recall = true_positive_classes / (false_negative_classes+true_positive_classes)

log_string('Semantic Segmentation Precision: {}'.format(precision))
log_string('Semantic Segmentation Recall: {}'.format(recall))
log_string('Semantic Segmentation F1-score: {}'.format(2*precision*recall/(precision+recall)))
log_string('Semantic Segmentation IoU: {}'.format( np.array(iou_list)))

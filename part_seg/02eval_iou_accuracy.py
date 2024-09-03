import os
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

NUM_CLASSES = 2
MAX_POINTS = 4096

pred_data_label_filenames = []
file_name = os.path.join(ROOT_DIR, "part_seg/output_filelist.txt")
pred_data_label_filenames += [line.rstrip() for line in open(file_name)]

gt_label_filenames = [f.rstrip('pred\.txt') + 'gt.txt' for f in pred_data_label_filenames]

num_room = len(gt_label_filenames)

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

data_label = np.loadtxt(os.path.join(ROOT_DIR, pred_data_label_filenames[0]))
gt_label = np.loadtxt(os.path.join(ROOT_DIR, gt_label_filenames[0]))

plant_number = data_label.shape[0]//MAX_POINTS

for i in range(plant_number):
    start_index = i * MAX_POINTS
    end_index = (i+1) * MAX_POINTS
    pred_sem = data_label[start_index:end_index, -1].reshape(-1).astype(int)
    gt_sem = gt_label[start_index:end_index, -1].reshape(-1).astype(int)

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

LOG_FOUT = open('Epoch46.txt', 'w')
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

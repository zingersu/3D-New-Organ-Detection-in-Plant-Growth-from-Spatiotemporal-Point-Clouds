import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import random
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import part_dataset_all_normal

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 1]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default=os.path.join(ROOT_DIR, 'part_seg/log/best_model.ckpt'),
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='Epoch46', help='Log dir [default: log_eval]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--input_list_test', type=str, default='data/test_file_list.txt', help='Input data list file')
FLAGS = parser.parse_args()
import time

sum = 0

VOTE_NUM = 12

EPOCH_CNT = 0
count = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
# RATIO = FLAGS.ratio
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
NUM_CLASSES = 2
TEST_FILE_LIST = FLAGS.input_list_test

OUTPUT_DIR = os.path.join(LOG_DIR, 'test_results')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


test_file_list = provider.getDataFiles(os.path.join(ROOT_DIR, TEST_FILE_LIST))
test_data = []
test_sem = []
test_time_sem = []
for h5_filename_test in test_file_list:
    cur_data,cur_organ_sem, cur_time_sem = provider.load_h5_data_label_seg1(os.path.join(ROOT_DIR, h5_filename_test))
    test_data.append(cur_data)
    test_sem.append(cur_organ_sem)
    test_time_sem.append(cur_time_sem)

test_data = np.concatenate(test_data, axis=0)
test_sem = np.concatenate(test_sem, axis=0)
test_time_sem = np.concatenate(test_time_sem, axis=0)

test_time_sem = np.expand_dims(test_time_sem, axis=-1)
test_data = np.concatenate((test_data, test_time_sem), axis=-1)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model and loss")
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps, normal, seg = dataset[idxs[i + start_idx]]
        batch_data[i, :, 0:3] = ps
        batch_data[i, :, 3:6] = normal
        batch_label[i, :] = seg
    return batch_data, batch_label


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (test_data.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    current_data = test_data
    current_label = np.squeeze(test_sem).astype(np.uint8)

    seg_classes = {'plant': [0,1]}
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}   # {0:'plant1', 1:'plant1', ..., 4：'plant3', 5:'plant4'}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    out_data_label_filename = 'out_pred.txt'
    out_data_label_filename = os.path.join(OUTPUT_DIR, out_data_label_filename)
    out_gt_label_filename = 'out_gt.txt'
    out_gt_label_filename = os.path.join(OUTPUT_DIR, out_gt_label_filename)
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            log_string('%03d/%03d' % (batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(test_data.shape[0], (batch_idx + 1) * BATCH_SIZE)
        cur_batch_size = end_idx - start_idx
        cur_batch_data = current_data[start_idx:end_idx, :, :]
        cur_batch_label = current_label[start_idx:end_idx]
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label

        # ---------------------------------------------------------------------
        loss_val = 0
        pred_val = np.zeros((BATCH_SIZE, NUM_POINT, NUM_CLASSES))
        for _ in range(VOTE_NUM):
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training}
            start = time.clock()
            global count
            count += 1
            temp_loss_val, temp_pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            global sum
            sum += time.clock() - start
            loss_val += temp_loss_val
            pred_val += temp_pred_val
        loss_val /= float(VOTE_NUM)
        # ---------------------------------------------------------------------

        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

        for i in range(cur_batch_size):
            for j in range(cur_pred_val.shape[1]):
                fout_data_label.write('%f %f %f %d %d\n' % (
                    batch_data[i, j, 0], batch_data[i, j, 1], batch_data[i, j, 2], batch_data[i, j, 3], cur_pred_val[i, j]))
                fout_gt_label.write('%f %f %f %d %d\n' % (
                    batch_data[i, j, 0], batch_data[i, j, 1], batch_data[i, j, 2], batch_data[i, j, 3], batch_label[i, j]))

        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (cur_batch_size * NUM_POINT)
        if cur_batch_size == BATCH_SIZE:
            loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (cur_batch_label == l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = cur_batch_label[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    fout_data_label.close()
    fout_gt_label.close()

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    sum_ious=0    
    for i in shape_ious.keys():
            sum_ious=sum_ious+shape_ious[i]
    mean_shape_ious =sum_ious/ len(shape_ious)
    log_string('eval mean loss: %f' % (loss_sum / float(test_data.shape[0] / BATCH_SIZE)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f' % (cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))
    log_string('test time: %f' % (sum))


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
    print("sum time:", sum)

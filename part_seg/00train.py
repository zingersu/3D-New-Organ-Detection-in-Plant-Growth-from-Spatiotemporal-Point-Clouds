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
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default= 4096, help='Point Number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default= 200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default= 8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.002]')
parser.add_argument('--momentum', type=float, default=0.90, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=30000, help='Decay step for lr decay [default: 30000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--input_list_train', type=str,default="data/train_file_list.txt",help='Input data list file')
parser.add_argument('--input_list_test', type=str,default="data/test_file_list.txt",help='Input data list file')
parser.add_argument('--restore_model', type=str, default='log/', help='Pretrained model')
FLAGS = parser.parse_args()
import time

sum1 = 0
EPOCH_CNT = 0
best_indicators = 0.0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

TRAINING_FILE_LIST = FLAGS.input_list_train
TEST_FILE_LIST = FLAGS.input_list_test
PRETRAINED_MODEL_PATH = FLAGS.restore_model
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
# RATIO = FLAGS.ratio

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 2


def convert_label_to_one_hot(labels):
  label_one_hot = np.zeros((labels.shape[0], labels.shape[1], NUM_CLASSES))
  for idx in range(labels.shape[0]):
      for jdx in range(labels.shape[1]):
        label_one_hot[idx, jdx, int(labels[idx, jdx])] = 1
      return label_one_hot


# Load train data
train_file_list = provider.getDataFiles(os.path.join(ROOT_DIR, TRAINING_FILE_LIST))
train_data = []
train_sem = []
train_time_sem = []
for h5_filename in train_file_list:
    cur_data,cur_organ_sem, cur_time_sem = provider.load_h5_data_label_seg1(os.path.join(ROOT_DIR, h5_filename))
    train_data.append(cur_data)
    train_sem.append(cur_organ_sem)
    train_time_sem.append(cur_time_sem)
train_data = np.concatenate(train_data, axis=0)
train_sem = np.concatenate(train_sem, axis=0)

train_time_sem = np.concatenate(train_time_sem, axis=0)
train_time_sem = np.expand_dims(train_time_sem, axis=-1)
train_data = np.concatenate((train_data, train_time_sem), axis=-1)

# Load test data
test_file_list = provider.getDataFiles(os.path.join(ROOT_DIR, TEST_FILE_LIST))
test_data = []
test_sem = []
test_time_sem = []
for h5_filename_test in test_file_list:
    cur_data, cur_organ_sem, cur_time_sem = provider.load_h5_data_label_seg1(os.path.join(ROOT_DIR, h5_filename_test))
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


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, cat_num=NUM_CLASSES)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            load_var_list = [v for v in tf.all_variables() if ('sem_' not in v.name)]
            loader = tf.train.Saver(load_var_list, sharded=True)
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=MAX_EPOCH)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            log_string("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            log_string("Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 2 == 0 or epoch == (MAX_EPOCH - 1):
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))
                log_string("Model saved in file: %s" % save_path)


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


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    current_data,current_sem ,shuffled_idx = provider.shuffle_data(train_data[:, :, :],train_sem)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        current_data[start_idx:end_idx, :, :] = provider.jitter_point_cloud(current_data[start_idx:end_idx, :, :])
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_sem[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        start = time.clock()
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        global sum1
        sum1 += time.clock() - start
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_sem[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val

        if (batch_idx + 1) % 100 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 100))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
    log_string('train time: %f' % (sum1))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    global best_indicators
    is_training = False
    num_batches = (test_data.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data
    current_label = np.squeeze(test_sem).astype(np.uint8)

    seg_classes = {'plant': [0,1]}
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 4))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    for batch_idx in range(num_batches):
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
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
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

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_shape_ious = sum(shape_ious.values()) / len(shape_ious)
    log_string('eval mean loss: %f' % (loss_sum / float(test_data.shape[0] / BATCH_SIZE)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f' % (cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))

    # find the best model
    best_saver = tf.train.Saver(max_to_keep=1)
    judge_indicators = (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)) + mean_shape_ious) / 2
    if judge_indicators >= best_indicators:
        best_indicators = judge_indicators
        save_path = best_saver.save(sess, os.path.join(LOG_DIR, 'best_model' + '.ckpt'))
        log_string("Model saved in file: %s" % save_path)

    EPOCH_CNT += 1
    return total_correct / float(total_seen)


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
    print("sum time:", sum1)

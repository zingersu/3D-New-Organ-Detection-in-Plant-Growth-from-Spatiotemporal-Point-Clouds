import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from transform_nets import input_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    """ tf.placeholder()函数创建一个占位符. 占位符是在TensorFlow图中的一个容器（节点）, 可以在运行图的时候向这个容器中填充数据, 占位符用于批量接受外界输入, tf.float32表示这个占位符中的元素都是32位浮点数
    这是占位符的形状（shape）, 占位符中的数据会被要求符合这个形状; 即tf.placeholder()表示TensorFlow中创建一个占位符（placeholder）的语句"""
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    # input_label_phs = tf.placeholder(tf.int32, shape=(batch_size, NUM_CLASSES))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, bn=True, cat_num=None):
    data_format = 'NHWC'
    """ 语义分割PointNet, 输入为 BxNx3, 输出为Bxnum_class"""
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, :3]

    input_image = tf.expand_dims(point_cloud, -1)

    k = 20

    adj = tf_util.pairwise_distance(l0_points)    # 返回的结果是一个形状为(5, 4096, 4096)的数组, 其中每个元素是对应点对之间的平方欧氏距离
    nn_idx = tf_util.dg_knn(adj, k=k)  # (batch, num_points, k); 在 adj_matrix 的最后一个维度上找出其中最小的 k 个值的索引, 具体实现参照tf_util中的dg_knn()函数
    edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=4, is_dist=True)
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    input_image = tf.expand_dims(point_cloud_transformed, -1)
    adj = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.dg_knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)


    out1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv1d', bn_decay=bn_decay,
                          data_format=data_format)

    out2 = tf_util.conv2d(out1, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv2d', bn_decay=bn_decay,
                          data_format=data_format)


    net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.dg_knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)
    
    out3 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv3d', bn_decay=bn_decay,
                          data_format=data_format)

    out4 = tf_util.conv2d(out3, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv4d', bn_decay=bn_decay,
                          data_format=data_format)

    net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_2)
    nn_idx = tf_util.dg_knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)
    
    out5 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv5d', bn_decay=bn_decay,
                          data_format=data_format)

    # out6 = tf_util.conv2d(out5, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training, weight_decay=weight_decay,
    #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

    net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv7d', bn_decay=bn_decay,
                          data_format=data_format)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    # one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    # one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],
    #                                       padding='VALID', stride=[1, 1],
    #                                       bn=True, is_training=is_training,
    #                                       scope='one_hot_label_expand', bn_decay=bn_decay)
    # out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv1')
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv2')
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp2')
    end_points['feats'] = net
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='seg/conv3')
    net = tf_util.conv2d(net, 2, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, bn=False, scope='seg/conv4')
    net = tf.reshape(net, [batch_size, num_point, 2])

    return net, end_points



def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    #
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)

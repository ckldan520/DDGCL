from config import cfg
from warpper import Unsupervised_graph
from graph_data_process import NeighborFinder, count_neighbors
from alive_progress import alive_bar
from sklearn.metrics import average_precision_score, roc_auc_score
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import sys


def var_filter(var_list, filter_keywords, last_layers=[0]):
    result_var = []
    for var in var_list:
        for layer in last_layers:
            kw = filter_keywords[layer]
            if kw in var.name:
                result_var.append(var)
                break
        else:
            continue

    return result_var


if __name__ == '__main__':

    if len(sys.argv) > 1:
        DATA = sys.argv[1]
    else:
        DATA = cfg.data

    g_df = pd.read_csv('./dataset/ml_{}.csv'.format(DATA))
    e_feat = np.load('./dataset/ml_{}.npy'.format(DATA))
    n_feat = np.load('./dataset/ml_{}_node.npy'.format(DATA))

    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
    # Divide the data set by time
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    # Training set
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]
    # Validation set
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]
    # Test set
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]

    max_idx = max(src_l.max(), dst_l.max())
    cfg.update_config('max_num_nodes', max_idx + 1)

    # graph only contains the training data (with 10% nodes removal)
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=False)

    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=False)

    """ """
    input_node1 = tf.placeholder(tf.int32, [None])
    input_node2 = tf.placeholder(tf.int32, [None])
    input_time = tf.placeholder(tf.float32, [None])
    input_cl_time = tf.placeholder(tf.float32, [None])
    input_label_node = tf.placeholder(tf.float32, [None])
    input_neighbor_l1 = tf.placeholder(tf.int32, [None, 3, cfg.neigh_num])
    input_neighbor_l2 = tf.placeholder(tf.int32, [None, 3, cfg.neigh_num * cfg.neigh_num])
    input_neighbor_time_l1 = tf.placeholder(tf.float32, [None, 3, cfg.neigh_num])
    input_neighbor_time_l2 = tf.placeholder(tf.float32, [None, 3, cfg.neigh_num * cfg.neigh_num])
    input_edge_l1 = tf.placeholder(tf.float32, [None, 3, cfg.neigh_num, cfg.input_feat_dim])
    input_edge_l2 = tf.placeholder(tf.float32, [None, 3, cfg.neigh_num * cfg.neigh_num, cfg.input_feat_dim])
    input_train_flag = tf.placeholder(tf.bool)

    instan_graph = Unsupervised_graph(cfg)
    class_logit, contrast_loss = instan_graph.call(input_node1, input_node2, input_time, input_neighbor_l1,
                                                   input_neighbor_l2,
                                                   input_neighbor_time_l1, input_neighbor_time_l2, input_edge_l1,
                                                   input_edge_l2, input_cl_time, input_train_flag)

    prob_node = tf.nn.softmax(class_logit)
    x_label = tf.cast(input_label_node, tf.int32)
    supervision_loss = 1 * tf.losses.sparse_softmax_cross_entropy(x_label, class_logit)
    tv = tf.trainable_variables()
    tv = var_filter(tv, filter_keywords=['classifier'], last_layers=[0])
    regularization_cost = 1e-6 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    total_loss = supervision_loss + contrast_loss + regularization_cost

    gs = tf.train.get_or_create_global_step()
    gs = tf.assign_add(gs, 1)
    total_steps = len(train_src_l) // cfg.batchsize * 2 * cfg.epochs
    new_lr = cfg.lr * 0.5 * (1 + tf.cos(gs / total_steps * np.pi))

    train_op = tf.train.AdamOptimizer(new_lr).minimize(total_loss)

    initialize_op = tf.variables_initializer([instan_graph.node_features, instan_graph.net.queue,
                                              instan_graph.net.queue_time, instan_graph.net.queue_ptr])
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        for i_step in range(10):
            print('total org times: %d' % i_step)
            step = 0
            max_valid_auc = 0
            count_stoppong_step = 0
            sess.run(tf.global_variables_initializer())

            while step < cfg.epochs:
                print('epoch: %d in the time' % step)
                step += 1

                sess.run([initialize_op])
                num_instance = len(train_src_l)
                num_batch = math.ceil(num_instance / cfg.batchsize)
                ave_loss = 0
                ave_t1_loss = 0
                ave_t2_loss = 0
                ave_t3_loss = 0
                for k in range(num_batch):
                    s_idx = k * cfg.batchsize
                    e_idx = min(num_instance - 1, s_idx + cfg.batchsize)
                    b = e_idx - s_idx
                    # construct subgraph
                    src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
                    ts_l_cut = train_ts_l[s_idx:e_idx]
                    label_l_cut = train_label_l[s_idx:e_idx]  # 节点分类的label
                    # ----------sample of constractive learning
                    ngh1_node_batch, ngh1_edge_batch, ngh1_t_batch, ngh2_node_batch, ngh2_edge_batch, ngh2_t_batch, cl_time = \
                        count_neighbors(train_ngh_finder, src_l_cut, dst_l_cut, ts_l_cut, cfg.neigh_num)

                    ngh1_edge_feature = e_feat[ngh1_edge_batch.flatten(), :].reshape([b, 3, cfg.neigh_num, -1])
                    ngh2_edge_feature = e_feat[ngh2_edge_batch.flatten(), :].reshape(
                        [b, 3, cfg.neigh_num * cfg.neigh_num, -1])

                    _, out_loss, out_t1_loss, out_t2_loss, out_t3_loss, out_learn = sess.run(
                        [train_op, total_loss, supervision_loss, contrast_loss, regularization_cost, new_lr],
                        feed_dict={
                            input_node1: src_l_cut, input_node2: dst_l_cut,
                            input_time: ts_l_cut,
                            input_cl_time: cl_time,
                            input_label_node: label_l_cut,
                            input_neighbor_l1: ngh1_node_batch, input_neighbor_l2: ngh2_node_batch,
                            input_neighbor_time_l1: ngh1_t_batch, input_neighbor_time_l2: ngh2_t_batch,
                            input_edge_l1: ngh1_edge_feature, input_edge_l2: ngh2_edge_feature,
                            input_train_flag: True})

                    ave_loss += out_loss

                    ave_t1_loss += out_t1_loss
                    ave_t2_loss += out_t2_loss
                    ave_t3_loss += out_t3_loss

                    if (k + 1) % (100) == 0:
                        ave_loss = ave_loss / (100)
                        ave_t1_loss = ave_t1_loss / (100)
                        ave_t2_loss = ave_t2_loss / (100)
                        ave_t3_loss = ave_t3_loss / (100)

                        print('step: %d   ——  train_loss: %.4f ( %.4f + %.4f + %.4f ) —— learning_rate: %.6f' % (
                            (k + 1) // (100), ave_loss, ave_t1_loss, ave_t2_loss, ave_t3_loss, out_learn))
                        ave_loss = 0
                        ave_t1_loss = 0
                        ave_t2_loss = 0
                        ave_t3_loss = 0

                # validation phase
                print('<<<<<<<<<<<<<<Validation phase>>>>>>>>>>')
                num_instance = len(val_src_l)
                num_batch = math.ceil(num_instance / cfg.batchsize)
                val_probs = []
                val_lables = []
                with alive_bar(num_batch) as bar:
                    for k in range(num_batch):
                        bar()
                        s_idx = k * cfg.batchsize
                        e_idx = min(num_instance - 1, s_idx + cfg.batchsize)
                        src_l_cut = val_src_l[s_idx:e_idx]
                        dst_l_cut = val_dst_l[s_idx:e_idx]
                        ts_l_cut = val_ts_l[s_idx:e_idx]
                        label_l_cut = val_label_l[s_idx:e_idx]
                        b = e_idx - s_idx

                        ngh1_node_batch, ngh1_edge_batch, ngh1_t_batch, ngh2_node_batch, ngh2_edge_batch, ngh2_t_batch, cl_time = \
                            count_neighbors(full_ngh_finder, src_l_cut, dst_l_cut, ts_l_cut, cfg.neigh_num)

                        ngh1_edge_feature = e_feat[ngh1_edge_batch.flatten(), :].reshape([b, 3, cfg.neigh_num, -1])
                        ngh2_edge_feature = e_feat[ngh2_edge_batch.flatten(), :].reshape(
                            [b, 3, cfg.neigh_num * cfg.neigh_num, -1])

                        out_prob, = sess.run([prob_node],
                                             feed_dict={input_node1: src_l_cut, input_node2: dst_l_cut,
                                                        input_time: ts_l_cut,
                                                        input_cl_time: cl_time,
                                                        input_neighbor_l1: ngh1_node_batch,
                                                        input_neighbor_l2: ngh2_node_batch,
                                                        input_neighbor_time_l1: ngh1_t_batch,
                                                        input_neighbor_time_l2: ngh2_t_batch,
                                                        input_edge_l1: ngh1_edge_feature,
                                                        input_edge_l2: ngh2_edge_feature,
                                                        input_train_flag: False})

                        val_probs.extend(out_prob)
                        val_lables.extend(label_l_cut)

                print("Validation Result")
                val_probs_mat = np.array(val_probs).reshape(-1, 2)
                val_labels_mat = np.array(val_lables).ravel()
                val_auc_count = roc_auc_score(val_labels_mat, val_probs_mat[:, 1])
                print(f'auc: {val_auc_count:.4f}')

                # test phase
                print('<<<<<<<<<<<<<<Test phase>>>>>>>>>>')
                num_instance = len(test_src_l)
                num_batch = math.ceil(num_instance / cfg.batchsize)
                test_probs = []
                test_lables = []
                with alive_bar(num_batch) as bar:
                    for k in range(num_batch):
                        bar()
                        s_idx = k * cfg.batchsize
                        e_idx = min(num_instance - 1, s_idx + cfg.batchsize)
                        src_l_cut = test_src_l[s_idx:e_idx]
                        dst_l_cut = test_dst_l[s_idx:e_idx]
                        ts_l_cut = test_ts_l[s_idx:e_idx]
                        label_l_cut = test_label_l[s_idx:e_idx]
                        b = e_idx - s_idx

                        ngh1_node_batch, ngh1_edge_batch, ngh1_t_batch, ngh2_node_batch, ngh2_edge_batch, ngh2_t_batch, cl_time = \
                            count_neighbors(full_ngh_finder, src_l_cut, dst_l_cut, ts_l_cut, cfg.neigh_num)

                        ngh1_edge_feature = e_feat[ngh1_edge_batch.flatten(), :].reshape([b, 3, cfg.neigh_num, -1])
                        ngh2_edge_feature = e_feat[ngh2_edge_batch.flatten(), :].reshape(
                            [b, 3, cfg.neigh_num * cfg.neigh_num, -1])
                        out_prob, = sess.run([prob_node],
                                             feed_dict={input_node1: src_l_cut, input_node2: dst_l_cut,
                                                        input_time: ts_l_cut,
                                                        input_cl_time: cl_time,
                                                        input_neighbor_l1: ngh1_node_batch,
                                                        input_neighbor_l2: ngh2_node_batch,
                                                        input_neighbor_time_l1: ngh1_t_batch,
                                                        input_neighbor_time_l2: ngh2_t_batch,
                                                        input_edge_l1: ngh1_edge_feature,
                                                        input_edge_l2: ngh2_edge_feature,
                                                        input_train_flag: False})

                        test_probs.extend(out_prob)
                        test_lables.extend(label_l_cut)

                print("Test Result")
                test_probs_mat = np.array(test_probs).reshape(-1, 2)
                test_labels_mat = np.array(test_lables).ravel()
                test_auc_count = roc_auc_score(test_labels_mat, test_probs_mat[:, 1])
                print(f'auc: {test_auc_count:.4f}')

                # Early stopping
                max_valid_auc = 0
                count_stoppong_step = 0
                if val_auc_count > max_valid_auc:
                    max_valid_auc = val_auc_count
                    count_stoppong_step = step

                if step - count_stoppong_step > 5:
                    break

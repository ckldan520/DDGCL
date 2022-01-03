import tensorflow as tf
from moco_builder import Moco


class Unsupervised_graph():
    def __init__(self, config):
        self.dims = config.dims
        self.init_nodes = config.max_num_nodes
        self.batch_size = config.batchsize

        with tf.device('/cpu:0'):
            self.node_features = tf.Variable(tf.random_normal([self.init_nodes, self.dims]), dtype=tf.float32,
                                             trainable=False, name='node_features')
        # layer_norm params
        self.net = Moco(config)

    def cal_class_logit(self, source_node, target_node, prob, train_flag):
        e = tf.concat([source_node, target_node], axis=-1)
        e = tf.layers.dense(e, units=1024, activation=tf.nn.leaky_relu, use_bias=True, name='classifier_conv1',
                            reuse=tf.AUTO_REUSE)
        e = tf.layers.dropout(e, rate=prob, training=train_flag)
        e = tf.layers.dense(e, units=512, activation=tf.nn.leaky_relu, use_bias=True, name='classifier_conv2',
                            reuse=tf.AUTO_REUSE)
        e = tf.layers.dropout(e, rate=prob, training=train_flag)
        e = tf.layers.dense(e, units=2, name='classifier_out_layer', reuse=tf.AUTO_REUSE)
        return e

    def feature_extract(self, feature, id):
        return tf.nn.embedding_lookup(feature, id)

    def call(self, source_node, target_node, current_time, neighbor_l1, neighbor_l2, neighbor_time_l1, neighbor_time_l2,
             edge_l1, edge_l2, cl_time, train_flag):
        '''
        :param self:
        :param source_node: [b]
        :param target_node: [b]
        :param current_time: [b]
        :param neighbor_l1: [b, 3, neigh]
        :param neighbor_l2: [b, 3, neigh*neigh]
        :param neighbor_time_l1: [b, 3, neigh]
        :param neighbor_time_l2: [b, 3, neigh*neigh]
        :param edge_l1: [b, 3, neigh, 172]
        :param edge_l2: [b, 3, neigh*neigh, 172]
        :return: logit [b, 2]
        '''

        src_node_state = self.feature_extract(self.node_features, source_node)
        des_node_state = self.feature_extract(self.node_features, target_node)
        neigh_l1_state = self.feature_extract(self.node_features, neighbor_l1)
        neigh_l2_state = self.feature_extract(self.node_features, neighbor_l2)

        src_query = tf.tuple([src_node_state, current_time,
                              tf.gather(neigh_l1_state, indices=0, axis=1),
                              tf.gather(neigh_l2_state, indices=0, axis=1),
                              tf.gather(neighbor_time_l1, indices=0, axis=1),
                              tf.gather(neighbor_time_l2, indices=0, axis=1),
                              tf.gather(edge_l1, indices=0, axis=1), tf.gather(edge_l2, indices=0, axis=1)])

        des_query = tf.tuple([des_node_state, current_time,
                              tf.gather(neigh_l1_state, indices=1, axis=1),
                              tf.gather(neigh_l2_state, indices=1, axis=1),
                              tf.gather(neighbor_time_l1, indices=1, axis=1),
                              tf.gather(neighbor_time_l2, indices=1, axis=1),
                              tf.gather(edge_l1, indices=1, axis=1), tf.gather(edge_l2, indices=1, axis=1)])

        key_query = tf.tuple([src_node_state, cl_time,
                              tf.gather(neigh_l1_state, indices=2, axis=1),
                              tf.gather(neigh_l2_state, indices=2, axis=1),
                              tf.gather(neighbor_time_l1, indices=2, axis=1),
                              tf.gather(neighbor_time_l2, indices=2, axis=1),
                              tf.gather(edge_l1, indices=2, axis=1),
                              tf.gather(edge_l2, indices=2, axis=1)])

        # constrastive learning loss
        constract_loss = self.net.constract_learning(src_query, key_query)

        # classification loss
        src_embed = self.net.encode(src_query)
        des_embed = self.net.encode(des_query)
        class_logit = self.cal_class_logit(src_embed, des_embed, 0.1, train_flag)

        return class_logit, constract_loss

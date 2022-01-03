import tensorflow as tf
import numpy as np

from transformer import Attention_Net

class Base_encoder():
    def __init__(self, config, name_space):
        self.dims = config.dims
        self.input_feat_dim = config.input_feat_dim
        self.name_scope = name_space
        self.transformer_net_l1 = Attention_Net(config)
        self.transformer_net_l2 = Attention_Net(config)

    def aggre_fun(self, Trans_model, node_state, node_time_embedding, neigh_state, neigh_delta_embedding, neigh_edge,
                  name_space='name_space'):
        with tf.variable_scope(self.name_scope + name_space, reuse=tf.AUTO_REUSE):
            new_node_state = Trans_model.model(node_state, node_time_embedding, neigh_state,
                                               neigh_delta_embedding, neigh_edge)
        return new_node_state

    def time_coding(self, delta_t, h_dim, scope=''):
        time_dim = h_dim // 2
        init_value = 1 / 10 ** np.linspace(0, 1.5, time_dim)
        with tf.variable_scope(self.name_scope + 'time_encoding', reuse=tf.AUTO_REUSE):
            self.basis_freq = tf.get_variable(
                'time_w', [time_dim],
                initializer=tf.constant_initializer(init_value))

        inputs = tf.log(delta_t + 1)
        map_ts = tf.matmul(tf.reshape(inputs, [-1, 1]),
                           tf.reshape(self.basis_freq, [1, -1]))
        harmonic = tf.concat([tf.cos(map_ts), tf.sin(map_ts)], axis=-1)
        return harmonic

    def encode(self, query):
        '''
                :param source_state: [b, dim]
                :param current_time: [b]
                :param neigh_l1_state: [b, neigh, dim]
                :param neigh_l2_state: [b,neigh*neigh, dim]
                :param neigh_l1_time: [b, neigh]
                :param neigh_l2_time: [b, neigh*neigh]
                :param edge_l1: [b,neigh, 172]
                :param edge_l2: [b, neigh*neigh, 172]
                :return:
                '''
        source_state, current_time, neigh_l1_state, neigh_l2_state, neigh_l1_time, neigh_l2_time, \
        edge_l1, edge_l2 = query

        b = tf.shape(current_time)[0]
        t = tf.shape(neigh_l1_state)[1]

        neigh_l2_delta = tf.expand_dims(current_time, axis=1) - neigh_l2_time
        neigh_l1_delta = tf.expand_dims(current_time, axis=1) - neigh_l1_time
        neigh_l1_delta_embedding = self.time_coding(tf.reshape(neigh_l1_delta, [-1, 1]), self.dims)
        neigh_l1_delta_embedding = tf.reshape(neigh_l1_delta_embedding, [b, -1, self.dims])  # [b, neigh, dims]
        neigh_l2_delta_embedding = self.time_coding(tf.reshape(neigh_l2_delta, [-1, 1]), self.dims)
        neigh_l2_delta_embedding = tf.reshape(neigh_l2_delta_embedding, [b, -1, self.dims])  # [b, neigh * neigh, dims]

        node_time_embedding = self.time_coding(tf.zeros_like(current_time), self.dims)
        node_time_embedding = tf.reshape(node_time_embedding, [b, -1, self.dims])
        node_l1_time_embedding = self.time_coding(tf.zeros_like(tf.reshape(neigh_l1_delta, [-1, 1])), self.dims)
        node_l1_time_embedding = tf.reshape(node_l1_time_embedding, [b, -1, self.dims])

        # level 1 cal
        source_state = tf.expand_dims(source_state, axis=1)
        new_source_node = self.aggre_fun(self.transformer_net_l1, source_state, node_time_embedding, neigh_l1_state,
                                         neigh_l1_delta_embedding,
                                         edge_l1, name_space="att1")

        neigh_l1_state = tf.reshape(neigh_l1_state, [b * t, 1, self.dims])
        node_l1_time_embedding = tf.reshape(node_l1_time_embedding, [b * t, 1, self.dims])
        neigh_l2_state = tf.reshape(neigh_l2_state, [b * t, t, self.dims])
        neigh_l2_delta_embedding = tf.reshape(neigh_l2_delta_embedding, [b * t, t, self.dims])
        edge_l2 = tf.reshape(edge_l2, [b * t, t, self.input_feat_dim])
        new_l1_state = self.aggre_fun(self.transformer_net_l1, neigh_l1_state, node_l1_time_embedding, neigh_l2_state,
                                      neigh_l2_delta_embedding,
                                      edge_l2, name_space="att1")
        new_l1_state = tf.reshape(new_l1_state, [b, t, self.dims])

        # level2 embedding - att2
        new_source_node = self.aggre_fun(self.transformer_net_l2, new_source_node, node_time_embedding, new_l1_state,
                                         neigh_l1_delta_embedding, edge_l1, name_space="att2")

        return new_source_node[:, 0, :], current_time


class Moco():
    def __init__(self, config):
        self.queue_size = 512
        self.dims = config.dims
        self.temp = 0.3

        queue_init = tf.zeros([self.queue_size, self.dims])

        self.queue = tf.get_variable('queue', initializer=queue_init, trainable=False)
        self.queue_time = tf.get_variable('queue_time', initializer=tf.zeros([self.queue_size, 1]), trainable=False)
        self.queue_ptr = tf.get_variable(
            'queue_ptr',
            [], initializer=tf.zeros_initializer(),
            dtype=tf.int64, trainable=False)

        self.encoder_q = Base_encoder(config, 'encoder_q/')
        self.encoder_k = Base_encoder(config, 'encoder_k/')

    def push_queue(self, item, item_time):
        # queue: [k, d]
        # item: [b, d]
        batch_size = tf.shape(item, out_type=tf.int64)[0]
        end_queue_ptr = self.queue_ptr + batch_size
        inds = tf.range(self.queue_ptr, end_queue_ptr, dtype=tf.int64)
        with tf.control_dependencies([inds]):
            queue_ptr_update = tf.assign(self.queue_ptr, end_queue_ptr % self.queue_size)
        queue_update = tf.scatter_update(self.queue, inds, item)
        queue_time_update = tf.scatter_update(self.queue_time, inds, tf.reshape(item_time, [-1, 1]))
        return tf.group(queue_update, queue_time_update, queue_ptr_update)

    def momentum_update(self, m=0.999):
        update_ops = [tf.assign_add(kw, (qw - kw) * (1 - m))
                      for qw, kw in zip(tf.trainable_variables('encoder_q'), tf.trainable_variables('encoder_k'))]
        return update_ops

    def loss_cpc_debais(self, q_feat, k_feat, q_time, k_time):  # 主
        beta = 1.5
        tau = 1 / 100

        time_encoding_pos = self.encoder_q.time_coding(tf.reshape(q_time - k_time, [-1, 1]), self.dims)  # b, dims

        queue_time = tf.reshape(self.queue_time, [-1])
        time_encoding_neg = self.encoder_q.time_coding(tf.reshape(tf.reduce_mean(q_time) - queue_time, [-1, 1]),
                                                       self.dims)  # k, dims

        W_pos = tf.layers.dense(time_encoding_pos, units=self.dims * self.dims, activation=tf.nn.relu, use_bias=True,
                                name='W_prediction', reuse=tf.AUTO_REUSE)
        W_pos = tf.reshape(W_pos, [-1, self.dims, self.dims])  # n, d, d

        W_neg = tf.layers.dense(time_encoding_neg, units=self.dims * self.dims, activation=tf.nn.relu, use_bias=True,
                                name='W_prediction', reuse=tf.AUTO_REUSE)
        W_neg = tf.reshape(W_neg, [-1, self.dims, self.dims])  # k, d, d

        queue = self.queue

        cpc_pos = tf.map_fn(lambda x: tf.squeeze(tf.transpose(x[0]) @ tf.expand_dims(x[1], -1), axis=-1),
                            (W_pos, k_feat), dtype=tf.float32)  # n, d
        cpc_neg = tf.map_fn(lambda x: tf.squeeze(tf.transpose(x[0]) @ tf.expand_dims(x[1], -1), axis=-1),
                            (W_neg, queue), dtype=tf.float32)  # k, d
        cpc_pos = q_feat * cpc_pos
        cpc_neg = tf.einsum('nc,kc->nkc', q_feat, cpc_neg)

        l_pos_sim = tf.reshape(tf.reduce_mean(cpc_pos, -1), [-1, 1])  # n 1
        l_neg_sim = tf.reduce_mean(cpc_neg, -1)  # n k

        l_pos = tf.sigmoid(l_pos_sim)
        label1 = tf.ones_like(l_pos)
        loss_pos = tf.keras.losses.binary_crossentropy(label1, l_pos)

        loss_neg_org = tf.log(1 / (1 + tf.exp(l_neg_sim)))

        neg_weight = tf.nn.softmax(beta * l_neg_sim)
        bais = tf.reshape(tau * tf.log(1 / (1 + tf.exp(l_pos_sim))), [-1])
        loss_neg = - (1 / (1 - tau)) * (tf.reduce_sum(neg_weight * loss_neg_org, axis=1) - bais)

        loss = tf.concat([loss_pos, loss_neg], axis=0)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        return loss

    def constract_learning(self, query, key):
        q_feat, q_time = self.encoder_q.encode(query)
        k_feat, k_time = self.encoder_k.encode(key)
        k_feat = tf.stop_gradient(k_feat)

        q_feat = tf.math.l2_normalize(q_feat, axis=-1)
        k_feat = tf.math.l2_normalize(k_feat, axis=-1)

        loss = self.loss_cpc_debais(q_feat, k_feat, q_time, k_time)

        queue_push_op = self.push_queue(k_feat, k_time)
        with tf.control_dependencies(
                [queue_push_op]):
            loss = loss + 0

        update_ops = self.momentum_update()
        with tf.control_dependencies(update_ops):
            loss = loss + 0

        return loss

    def encode(self, query):
        q_feat, _ = self.encoder_q.encode(query)
        q_feat = tf.math.l2_normalize(q_feat, axis=-1)

        return q_feat
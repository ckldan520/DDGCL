import tensorflow as tf


def gelu(input_tensor):
    """
    Activation function GELU
    :param input_tensor: A tensor
    :return:
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def layer_normalization(inputs,
                        last_dims,
                        epsilon=1e-8,
                        scope="ln",
                        reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # inputs_shape = inputs.get_shape()

        beta = tf.Variable(tf.zeros([last_dims]))
        gamma = tf.Variable(tf.ones([last_dims]))
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta
    return outputs


class PreNorm(tf.keras.Model):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = layer_normalization
        self.fn = fn
        self.dims = dim

    def __call__(self, x, **kwargs):
        x = self.norm(x, self.dims, reuse=tf.AUTO_REUSE)
        return self.fn(x, **kwargs)


class SelfAttention(tf.keras.Model):
    def __init__(self, cfg, one_kv_head=False):
        super(SelfAttention, self).__init__()

        self.heads = 8
        self.dim_head = cfg.dims // self.heads
        self.scale = self.dim_head ** (-0.5)
        self.dims = cfg.dims
        self.feature_len = cfg.node_pro_length

        self.to_q = tf.keras.layers.Dense(self.dims, use_bias=False, name='transformer_to_q')

        kv_dim = self.dim_head if one_kv_head else self.dims
        self.to_kv = tf.keras.layers.Dense(kv_dim * 2, use_bias=False, name='transformer_to_kv')
        self.to_out = tf.keras.layers.Dense(self.dims, name='transformer_to_out')

    def __call__(self, x, seq=None, **kwargs):
        """
        :param x: [Batch, feature_len, dim]
        :param updata_flag:
        :param memories:  {mem, cmem}->[Batch, current_len, dim(same with x)]
        :param pos_emb:
        :param calc_memory:
        :param kwargs:
        :return:
        """
        h, dim_h = self.heads, self.dim_head
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        e = tf.shape(x)[2]

        pro_seq = None
        if seq is not None:
            pro_seq = seq

        q = self.to_q(x)
        # kv_input = tf.concat([pro_seq, x], axis=1)
        kv_input = pro_seq
        kv_input = tf.reshape(kv_input, [b, -1, self.dims * 3])
        kv_out = self.to_kv(kv_input)
        k, v = tf.split(kv_out, num_or_size_splits=2, axis=-1)

        q = self.merge_heads(q, dim_h)
        k = self.merge_heads(k, dim_h)
        v = self.merge_heads(v, dim_h)

        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, [b, t, self.dims])
        logits = self.to_out(out)

        return logits

    def merge_heads(self, x, dim):
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] // dim, dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        return x


class FeedForward(tf.keras.Model):
    def __init__(self, dim, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.w1 = tf.keras.layers.Dense(dim * mult * 1, name='transformer_ff_w1')
        self.act = gelu
        # self.dropout = tf.keras.layers.Dropout(dropout)
        self.w2 = tf.keras.layers.Dense(dim, name='transformer_ff_w2')

    def __call__(self, x, **kwargs):
        x = self.w1(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.w2(x)

        return x



class Attention_Net():
    def __init__(self, cfg, one_kv_head=False, name=''):
        self.dims = cfg.dims
        self.net_depth = 1

        self.attn_layers = [PreNorm(self.dims * 3, SelfAttention(cfg, one_kv_head))]
        self.ff_layers = [PreNorm(self.dims, FeedForward(self.dims))]

    def model(self, src, src_t, seq, seq_t, seq_e):
        src_e = tf.zeros_like(src)
        x = tf.concat([src, src_e, src_t], axis=2)

        seq_e_todim = self.to_model_dim(seq_e)
        seq = tf.concat([seq, seq_e_todim, seq_t], axis=2)

        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            x = attn(x, seq=seq)
            x = ff(x)

        out = src + x

        return out

    def to_model_dim(self, x):
        """
        :param x: [Batch, ?]
        :return:  [Batch, 1, dims]
        """
        x = tf.layers.dense(x, units=self.dims, reuse=tf.AUTO_REUSE, name='Transformer_modeldim')
        x = tf.nn.relu(x, name='Transformer_modeldim_relu')
        return x
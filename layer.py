import tensorflow as tf
import pandas as pd
import numpy as np
from utils import *
from scipy.sparse import diags

seed = 6


class ib_talinear(tf.keras.layers.Layer):
    def __init__(self, pre_stamps, graph_dim, stamps, emb_dim, name="ib_talinear",
                 **kwargs):
        super(ib_talinear, self).__init__(name=name, **kwargs)
        self.pre_stamps = pre_stamps
        self.graph_dim = graph_dim
        self.emb_dim = emb_dim
        self.stamps = stamps
        self.stw = self.add_weight(name=self.name + '_stw', shape=[stamps, emb_dim, pre_stamps, emb_dim],
                                   initializer=tf.keras.initializers.he_normal(seed))
        self.stb = self.add_weight(name=self.name + '_stb',
                                   shape=[stamps, 1, emb_dim],
                                   initializer=tf.keras.initializers.he_normal(seed))

        self.w = self.add_weight(name=self.name + '_w',
                                 shape=[pre_stamps, emb_dim, pre_stamps, emb_dim],
                                 initializer=tf.keras.initializers.he_normal(seed))
        self.b = self.add_weight(name=self.name + '_b',
                                 shape=[pre_stamps, 1, emb_dim],
                                 initializer=tf.keras.initializers.he_normal(seed))

    def call(self, inputs, future_date):
        # future = future_date + future_week * self.stamps
        weight = tf.gather(self.stw, future_date)
        bias = tf.gather(self.stb, future_date)
        x = tf.einsum('btge, bteif->bigf', inputs, weight)
        d = x + bias

        x = tf.einsum('btge, teif->bigf', inputs, self.w)
        w = x + self.b

        return w + d

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"pre_stamps": self.pre_stamps, "graph_dim": self.graph_dim, "stamps": self.stamps,
                  "emb_dim": self.emb_dim}
        base_config = super(ib_talinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class mulit_time_aware_layers(tf.keras.layers.Layer):
    def __init__(self, pre_stamps, graph_dim, stamps, layer_num, output_dim, name="mulit_linear",
                 **kwargs):
        super(mulit_time_aware_layers, self).__init__(name=name, **kwargs)
        self.pre_stamps = pre_stamps
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.layers = []
        self.drop = []
        self.fusion = tf.keras.layers.Dense(output_dim, use_bias=False)
        for i in range(self.layer_num):
            self.layers.append(
                ib_talinear(pre_stamps, graph_dim, stamps, output_dim, name=self.name + "_layer_{}".format(i)))
            self.drop.append(
                tf.keras.layers.Dropout(0.5, noise_shape=[1, 1, 1, output_dim], name=self.name + "_drop_{}".format(i)))

        self.thta_In = self.add_weight(name=self.name + '_thtaInflow',
                                       shape=[pre_stamps, graph_dim, (layer_num) * (output_dim), output_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

    def call(self, inputs, future_date, graph_week_in_1, graph_week_in_2):
        outputs = []
        tmp = inputs
        for i in range(self.layer_num):
            hidden = tf.concat([tmp, graph_week_in_1, graph_week_in_2], axis=-1)
            hidden = self.fusion(hidden)
            tmp = self.layers[i](hidden, future_date)
            tmp = self.drop[i](tmp)
            outputs.append(tmp)
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.einsum('btge, tgef->btgf', outputs, self.thta_In)


        return outputs

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"pre_stamps": self.pre_stamps, "graph_dim": self.graph_dim, "stamps": self.stamps,
                  "layer_num": self.layer_num, "output_dim": self.output_dim}
        base_config = super(mulit_linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ib_talinear_no_date(tf.keras.layers.Layer):
    def __init__(self, pre_stamps, graph_dim, stamps, emb_dim, name="ib_talinear",
                 **kwargs):
        super(ib_talinear_no_date, self).__init__(name=name, **kwargs)
        self.pre_stamps = pre_stamps
        self.graph_dim = graph_dim
        self.emb_dim = emb_dim
        self.stamps = stamps
        self.stw = self.add_weight(name=self.name + '_stw', shape=[pre_stamps, emb_dim, pre_stamps, emb_dim],
                                   initializer=tf.keras.initializers.he_normal(seed))
        self.stb = self.add_weight(name=self.name + '_stb',
                                   shape=[pre_stamps, 1, emb_dim],
                                   initializer=tf.keras.initializers.he_normal(seed))

        self.w = self.add_weight(name=self.name + '_w',
                                 shape=[pre_stamps, emb_dim, pre_stamps, emb_dim],
                                 initializer=tf.keras.initializers.he_normal(seed))
        self.b = self.add_weight(name=self.name + '_b',
                                 shape=[pre_stamps, 1, emb_dim],
                                 initializer=tf.keras.initializers.he_normal(seed))

    def call(self, inputs):

        x = tf.einsum('btge, teif->bigf', inputs, self.stw)
        d = x + self.stb

        x = tf.einsum('btge, teif->bigf', inputs, self.w)
        w = x + self.b

        return w + d

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"pre_stamps": self.pre_stamps, "graph_dim": self.graph_dim, "stamps": self.stamps,
                  "emb_dim": self.emb_dim}
        base_config = super(ib_talinear_no_date, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class mulit_linear_no_date(tf.keras.layers.Layer):
    def __init__(self, pre_stamps, graph_dim, stamps, layer_num, output_dim, name="mulit_linear",
                 **kwargs):
        super(mulit_linear_no_date, self).__init__(name=name, **kwargs)
        self.pre_stamps = pre_stamps
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.layers = []
        self.drop = []
        self.fusion = tf.keras.layers.Dense(output_dim, use_bias=False)
        for i in range(self.layer_num):
            self.layers.append(
                ib_talinear_no_date(pre_stamps, graph_dim, stamps, output_dim, name=self.name + "_layer_{}".format(i)))
            self.drop.append(
                tf.keras.layers.Dropout(0.5, noise_shape=[1, 1, 1, output_dim], name=self.name + "_drop_{}".format(i)))

        self.thta_In = self.add_weight(name=self.name + '_thtaInflow',
                                       shape=[pre_stamps, graph_dim, (layer_num) * (output_dim), output_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

    def call(self, inputs, graph_day_in, graph_week_in):
        outputs = []
        tmp = inputs
        for i in range(self.layer_num):
            hidden = tf.concat([tmp, graph_day_in, graph_week_in], axis=-1)
            hidden = self.fusion(hidden)
            tmp = self.layers[i](hidden)
            tmp = self.drop[i](tmp)
            outputs.append(tmp)
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.einsum('btge, tgef->btgf', outputs, self.thta_In)

        return outputs

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"pre_stamps": self.pre_stamps, "graph_dim": self.graph_dim, "stamps": self.stamps,
                  "layer_num": self.layer_num, "output_dim": self.output_dim}
        base_config = super(mulit_linear_no_date, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class mulit_linear_no_time_emb(tf.keras.layers.Layer):
    def __init__(self, pre_stamps, graph_dim, stamps, layer_num, output_dim, name="mulit_linear",
                 **kwargs):
        super(mulit_linear_no_time_emb, self).__init__(name=name, **kwargs)

        self.pre_stamps = pre_stamps
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.layers = []
        self.drop = []
        self.fusion = tf.keras.layers.Dense(output_dim, use_bias=False)
        for i in range(self.layer_num):
            self.layers.append(
                ib_talinear(pre_stamps, graph_dim, stamps, output_dim, name=self.name + "_layer_{}".format(i)))
            self.drop.append(
                tf.keras.layers.Dropout(0.5, noise_shape=[1, 1, 1, output_dim], name=self.name + "_drop_{}".format(i)))

        self.thta_In = self.add_weight(name=self.name + '_thtaInflow',
                                       shape=[pre_stamps, graph_dim, (layer_num) * (output_dim), output_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

    def call(self, inputs, future_date):
        outputs = []
        tmp = inputs
        for i in range(self.layer_num):
            hidden = tmp
            hidden = self.fusion(hidden)
            tmp = self.layers[i](hidden, future_date)
            tmp = self.drop[i](tmp)
            outputs.append(tmp)
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.einsum('btge, tgef->btgf', outputs, self.thta_In)

        return outputs

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"pre_stamps": self.pre_stamps, "graph_dim": self.graph_dim, "stamps": self.stamps,
                  "layer_num": self.layer_num, "output_dim": self.output_dim}
        base_config = super(mulit_linear_no_time_emb, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class mulit_linear(tf.keras.layers.Layer):
    def __init__(self, pre_stamps, graph_dim, stamps, layer_num, output_dim, name="mulit_linear",
                 **kwargs):
        super(mulit_linear, self).__init__(name=name, **kwargs)
        self.pre_stamps = pre_stamps
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.layers = []
        self.drop = []
        self.fusion = tf.keras.layers.Dense(output_dim, use_bias=False)
        for i in range(self.layer_num):
            self.layers.append(
                ib_talinear(pre_stamps, graph_dim, stamps, output_dim, name=self.name + "_layer_{}".format(i)))
            self.drop.append(
                tf.keras.layers.Dropout(0.5, noise_shape=[1, 1, 1, output_dim], name=self.name + "_drop_{}".format(i)))

        self.thta_In = self.add_weight(name=self.name + '_thtaInflow',
                                       shape=[pre_stamps, graph_dim, (layer_num) * (output_dim), output_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

    def call(self, inputs, future_date, x_embdding, graph_week_in_1, graph_week_in_2):
        outputs = []
        tmp = inputs
        for i in range(self.layer_num):
            hidden = tf.concat([tmp, x_embdding, graph_week_in_1, graph_week_in_2], axis=-1)
            hidden = self.fusion(hidden)
            tmp = self.layers[i](hidden, future_date)
            tmp = self.drop[i](tmp)
            outputs.append(tmp)
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.einsum('btge, tgef->btgf', outputs, self.thta_In)

        return outputs

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"pre_stamps": self.pre_stamps, "graph_dim": self.graph_dim, "stamps": self.stamps,
                  "layer_num": self.layer_num, "output_dim": self.output_dim}
        base_config = super(mulit_linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class time_encoding(tf.keras.layers.Layer):
    ''' shift-invariant time encoding kernal

    inputs : [N, max_len]
    Returns: 3d float tensor which includes embedding dimension
    '''

    def __init__(self, time_dim):
        super(time_encoding, self).__init__()
        self.time_dim = time_dim
        self.effe_numits = time_dim // 2

        init_freq_base = np.linspace(0, 9, self.effe_numits).astype(np.float32)
        self.cos_freq_var = tf.Variable(initial_value=(1 / 10.0 ** init_freq_base).astype(np.float32),
                                        name='cos_freq_var')
        self.sin_freq_var = tf.Variable(initial_value=(1 / 10.0 ** init_freq_base).astype(np.float32),
                                        name='sin_freq_var')

        self.beta_var = tf.Variable(initial_value=np.ones(time_dim).astype(np.float32), name='beta_var')

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, [batch_size, seq_len, 1])
        inputs = tf.tile(inputs, [1, 1, self.effe_numits])
        #inputs = inputs.reshape(batch_size, seq_len, 1).repeat(1, 1, self.effe_numits)

        cos_freq_var = tf.reshape(self.cos_freq_var, [1, 1, self.effe_numits])
        sin_freq_var = tf.reshape(self.sin_freq_var, [1, 1, self.effe_numits])

        cos_feat = tf.sin(tf.multiply(inputs, cos_freq_var))
        sin_feat = tf.cos(tf.multiply(inputs, sin_freq_var))

        freq_feat = tf.concat((cos_feat, sin_feat), axis=-1)
        beta_var = tf.reshape(self.beta_var, [1, 1, self.time_dim])

        out = tf.multiply(freq_feat, beta_var)
        return out

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"time_dim": self.time_dim}
        base_config = super(time_encoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MLP(tf.keras.layers.Layer):
    def __init__(self, n_pre, graph_dim, layer_num, c,name="MLP", **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)
        self.n_pre = n_pre
        self.graph_dim = graph_dim
        self.layer_num = layer_num
        self.c = c
        self.weights1 = []
        self.bias1 = []
        self.drop = []
        self.weights2 = []
        self.bias2 = []

        for i in range(layer_num):
            self.weights1.append(self.add_weight(shape=[n_pre, graph_dim, graph_dim, n_pre],
                                                 initializer=tf.keras.initializers.glorot_normal(seed),
                                                 name=self.name + "_w1_{}".format(i)))

            self.weights2.append(self.add_weight(shape=[n_pre, graph_dim, graph_dim, n_pre],
                                                 initializer=tf.keras.initializers.glorot_normal(seed),
                                                 name=self.name + "_w2_{}".format(i)))

        #self.drop = tf.keras.layers.Dropout(0.5, noise_shape=[1, 1, c], name=self.name + "_drop")
        self.ln = tf.keras.layers.LayerNormalization(name='ln', axis=-2)

    def call(self, inputs):
        if self.layer_num > 0:
            for i in range(self.layer_num):
                hidden1 = tf.einsum('btgc, tgvi->bivc', inputs, self.weights1[i])
                tmp = tf.nn.relu(hidden1) + inputs
                hidden2 = tf.einsum('btgc, tgvi->bivc', tmp, self.weights2[i])
                inputs = self.ln(hidden2)

        return inputs

    def get_config(self):
        config = {"n_pre": self.n_pre, "graph_dim": self.graph_dim,
                  "layer_num": self.layer_num, "c": self.c}
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


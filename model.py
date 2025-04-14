from utils import *
from layer import *
from tensorflow import keras
import keras.backend as kb
import time
import pickle
import random
from numpy.linalg import norm

seed = 6


# input flow time patterns
# output flow space dependencies
class SGNN(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        if self.arg == 'HZ':
            self.graph_fe_in_week_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
            self.graph_fe_in_week_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
        else:
            if self.arg == 'SH':
                self.graph_fe_in_date_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_date_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
            else:
                self.graph_fe_in_date = self.add_weight(name=self.name + '_graph_fe_in1',
                                                        shape=[stamps, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_week = self.add_weight(name=self.name + '_graph_fe_in2',
                                                        shape=[days, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))

        self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_time_aware_layers(pre_stamps, graph_dim, stamps, l_num, f_c,
                                                        name=self.name + '_mulit_linear_out')

        self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

        self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, f_c, f_c],
                                        initializer=tf.keras.initializers.he_normal(seed))
        self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        b = inputs.get_shape()[0]
        past = inputs[:, :, 2, 0]
        future = inputs[:, :, 2, 0] + 4
        date = tf.concat([past, future], axis=1)
        date = tf.cast(date, dtype='int32')

        if b != None:
            average = []
            if self.total:
                for i in range(b):
                    average.append(tf.gather(self.mean['total'], date[i]))
            else:
                for i in range(b):
                    if inputs[i, 0, 1, 0] >= 5.0:
                        average.append(tf.gather(self.mean['weekends'], date[i]))
                    else:
                        average.append(tf.gather(self.mean['weekdays'], date[i]))
            average = tf.stack(average, axis=0)
            average = tf.cast(average, dtype='float32')
        else:
            average = tf.concat([inputs[:, :, 3:, :], inputs[:, :, 3:, :]], axis=1)
        # average -> b, 8, 80, 2

        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')
        week = tf.cast(inputs[:, :, 1, 0], dtype='int32')

        if self.arg == 'HZ':
            graph_in_1 = tf.gather(self.graph_fe_in_week_1, week)
            graph_in_2 = tf.gather(self.graph_fe_in_week_2, week)
        else:
            if self.arg == 'SH':
                graph_in_1 = tf.gather(self.graph_fe_in_date_1, date)
                graph_in_2 = tf.gather(self.graph_fe_in_date_2, date)
            else:
                graph_in_1 = tf.gather(self.graph_fe_in_date, date)
                graph_in_2 = tf.gather(self.graph_fe_in_week, week)

        delta_flow_4 = f - average[:, :4, :, :]

        # point-wise
        delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, date, graph_in_1, graph_in_2)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        #delta_flow_en = delta_flow_h
        E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        # 没有激活函数

        delta_flow_de = self.filter_Out(delta_flow)

        ouputs = tf.nn.relu(delta_flow_de + average[:, 4:, :, :])

        return ouputs

# no per
class SGNN_no_per(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN_no_per, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        if self.arg == 'HZ':
            self.graph_fe_in_week_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
            self.graph_fe_in_week_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
        else:
            if self.arg == 'SH':
                self.graph_fe_in_date_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_date_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
            else:
                self.graph_fe_in_date = self.add_weight(name=self.name + '_graph_fe_in1',
                                                        shape=[stamps, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_week = self.add_weight(name=self.name + '_graph_fe_in2',
                                                        shape=[days, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))

        self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_time_aware_layers(pre_stamps, graph_dim, stamps, l_num, f_c,
                                                        name=self.name + '_mulit_linear_out')

        self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

        self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, f_c, f_c],
                                        initializer=tf.keras.initializers.he_normal(seed))
        self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')
        week = tf.cast(inputs[:, :, 1, 0], dtype='int32')

        if self.arg == 'HZ':
            graph_in_1 = tf.gather(self.graph_fe_in_week_1, week)
            graph_in_2 = tf.gather(self.graph_fe_in_week_2, week)
        else:
            if self.arg == 'SH':
                graph_in_1 = tf.gather(self.graph_fe_in_date_1, date)
                graph_in_2 = tf.gather(self.graph_fe_in_date_2, date)
            else:
                graph_in_1 = tf.gather(self.graph_fe_in_date, date)
                graph_in_2 = tf.gather(self.graph_fe_in_week, week)

        delta_flow_4 = f

        # point-wise
        delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, date, graph_in_1, graph_in_2)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        #delta_flow_en = delta_flow_h
        E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        # 没有激活函数

        delta_flow_de = self.filter_Out(delta_flow)

        ouputs = tf.nn.relu(delta_flow_de)

        return ouputs
class SGNN_no_dyna(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN_no_dyna, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        if self.arg == 'HZ':
            self.graph_fe_in_week_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
            self.graph_fe_in_week_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                   shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
        else:
            if self.arg == 'SH':
                self.graph_fe_in_date_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_date_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
            else:
                self.graph_fe_in_date = self.add_weight(name=self.name + '_graph_fe_in1',
                                                        shape=[stamps, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_week = self.add_weight(name=self.name + '_graph_fe_in2',
                                                        shape=[days, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))

        self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_linear_no_date(pre_stamps, graph_dim, stamps, l_num, f_c,
                                                     name=self.name + '_mulit_linear_out')

        self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))
        self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, f_c, f_c],
                                        initializer=tf.keras.initializers.he_normal(seed))
        #self.drop = tf.keras.layers.Dropout(0.2, noise_shape=[1, 1, 1, f_c], name=self.name + "_drop")
        self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        b = inputs.get_shape()[0]
        past = inputs[:, :, 2, 0]
        future = inputs[:, :, 2, 0] + 4
        date = tf.concat([past, future], axis=1)
        date = tf.cast(date, dtype='int32')

        if b != None:
            average = []
            if self.total:
                for i in range(b):
                    average.append(tf.gather(self.mean['total'], date[i]))
            else:
                for i in range(b):
                    if inputs[i, 0, 1, 0] >= 5.0:
                        average.append(tf.gather(self.mean['weekends'], date[i]))
                    else:
                        average.append(tf.gather(self.mean['weekdays'], date[i]))
            average = tf.stack(average, axis=0)
            average = tf.cast(average, dtype='float32')
        else:
            average = tf.concat([inputs[:, :, 3:, :], inputs[:, :, 3:, :]], axis=1)
        # average -> b, 8, 80, 2

        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')
        week = tf.cast(inputs[:, :, 1, 0], dtype='int32')

        if self.arg == 'HZ':
            graph_in_1 = tf.gather(self.graph_fe_in_week_1, week)
            graph_in_2 = tf.gather(self.graph_fe_in_week_2, week)
        else:
            if self.arg == 'SH':
                graph_in_1 = tf.gather(self.graph_fe_in_date_1, date)
                graph_in_2 = tf.gather(self.graph_fe_in_date_2, date)
            else:
                graph_in_1 = tf.gather(self.graph_fe_in_date, date)
                graph_in_2 = tf.gather(self.graph_fe_in_week, week)

        delta_flow_4 = f - average[:, :4, :, :]

        # point-wise
        delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, graph_in_1, graph_in_2)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        # delta_flow_en = delta_flow_h
        E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        # 没有激活函数

        delta_flow_de = self.filter_Out(delta_flow)

        ouputs = tf.nn.relu(delta_flow_de + average[:, 4:, :, :])

        return ouputs


class SGNN_no_stamps_emb(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, t_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN_no_stamps_emb, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_linear_no_time_emb(pre_stamps, graph_dim, stamps, l_num, f_c,
                                                         name=self.name + '_mulit_linear_out')

        self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

        self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, f_c, f_c],
                                        initializer=tf.keras.initializers.he_normal(seed))
        self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        b = inputs.get_shape()[0]
        past = inputs[:, :, 2, 0]
        future = inputs[:, :, 2, 0] + 4
        date = tf.concat([past, future], axis=1)
        date = tf.cast(date, dtype='int32')

        if b != None:
            average = []
            if self.total:
                for i in range(b):
                    average.append(tf.gather(self.mean['total'], date[i]))
            else:
                for i in range(b):
                    if inputs[i, 0, 1, 0] >= 5.0:
                        average.append(tf.gather(self.mean['weekends'], date[i]))
                    else:
                        average.append(tf.gather(self.mean['weekdays'], date[i]))
            average = tf.stack(average, axis=0)
            average = tf.cast(average, dtype='float32')
        else:
            average = tf.concat([inputs[:, :, 3:, :], inputs[:, :, 3:, :]], axis=1)
        # average -> b, 8, 80, 2

        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')

        delta_flow_4 = f - average[:, :4, :, :]

        # point-wise
        delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, date)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        # delta_flow_en = delta_flow_h
        E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        # 没有激活函数

        delta_flow_de = self.filter_Out(delta_flow)

        ouputs = tf.nn.relu(delta_flow_de)

        return ouputs

class SGNN_no_hd(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, t_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN_no_hd, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        if self.arg == 'HZ':
            self.graph_fe_in_week_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
            self.graph_fe_in_week_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
        else:
            if self.arg == 'SH':
                self.graph_fe_in_date_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_date_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
            else:
                self.graph_fe_in_date = self.add_weight(name=self.name + '_graph_fe_in1',
                                                        shape=[stamps, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_week = self.add_weight(name=self.name + '_graph_fe_in2',
                                                        shape=[days, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))

        #self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_time_aware_layers(pre_stamps, graph_dim, stamps, l_num, 2,
                                                        name=self.name + '_mulit_linear_out')

        self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

        self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, 2, 2],
                                        initializer=tf.keras.initializers.he_normal(seed))
        #self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        b = inputs.get_shape()[0]
        past = inputs[:, :, 2, 0]
        future = inputs[:, :, 2, 0] + 4
        date = tf.concat([past, future], axis=1)
        date = tf.cast(date, dtype='int32')

        if b != None:
            average = []
            if self.total:
                for i in range(b):
                    average.append(tf.gather(self.mean['total'], date[i]))
            else:
                for i in range(b):
                    if inputs[i, 0, 1, 0] >= 5.0:
                        average.append(tf.gather(self.mean['weekends'], date[i]))
                    else:
                        average.append(tf.gather(self.mean['weekdays'], date[i]))
            average = tf.stack(average, axis=0)
            average = tf.cast(average, dtype='float32')
        else:
            average = tf.concat([inputs[:, :, 3:, :], inputs[:, :, 3:, :]], axis=1)
        # average -> b, 8, 80, 2

        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')
        week = tf.cast(inputs[:, :, 1, 0], dtype='int32')

        if self.arg == 'HZ':
            graph_in_1 = tf.gather(self.graph_fe_in_week_1, week)
            graph_in_2 = tf.gather(self.graph_fe_in_week_2, week)
        else:
            if self.arg == 'SH':
                graph_in_1 = tf.gather(self.graph_fe_in_date_1, date)
                graph_in_2 = tf.gather(self.graph_fe_in_date_2, date)
            else:
                graph_in_1 = tf.gather(self.graph_fe_in_date, date)
                graph_in_2 = tf.gather(self.graph_fe_in_week, week)

        delta_flow_4 = f - average[:, :4, :, :]

        # point-wise
        # delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, date, graph_in_1, graph_in_2)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        # delta_flow_en = delta_flow_h
        E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        #delta_flow_de = self.filter_Out(delta_flow)

        ouputs = tf.nn.relu(delta_flow + average[:, 4:, :, :])

        return ouputs

class SGNN_gnn(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN_gnn, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        if self.arg == 'HZ':
            self.graph_fe_in_week_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
            self.graph_fe_in_week_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
        else:
            if self.arg == 'SH':
                self.graph_fe_in_date_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_date_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
            else:
                self.graph_fe_in_date = self.add_weight(name=self.name + '_graph_fe_in1',
                                                        shape=[stamps, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_week = self.add_weight(name=self.name + '_graph_fe_in2',
                                                        shape=[days, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))

        self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_time_aware_layers(pre_stamps, graph_dim, stamps, l_num, f_c,
                                                        name=self.name + '_mulit_linear_out')

        #self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                      # initializer=tf.keras.initializers.he_normal(seed))

        #self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, f_c, f_c],
         #                               initializer=tf.keras.initializers.he_normal(seed))
        self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        b = inputs.get_shape()[0]
        past = inputs[:, :, 2, 0]
        future = inputs[:, :, 2, 0] + 4
        date = tf.concat([past, future], axis=1)
        date = tf.cast(date, dtype='int32')

        if b != None:
            average = []
            if self.total:
                for i in range(b):
                    average.append(tf.gather(self.mean['total'], date[i]))
            else:
                for i in range(b):
                    if inputs[i, 0, 1, 0] >= 5.0:
                        average.append(tf.gather(self.mean['weekends'], date[i]))
                    else:
                        average.append(tf.gather(self.mean['weekdays'], date[i]))
            average = tf.stack(average, axis=0)
            average = tf.cast(average, dtype='float32')
        else:
            average = tf.concat([inputs[:, :, 3:, :], inputs[:, :, 3:, :]], axis=1)
        # average -> b, 8, 80, 2

        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')
        week = tf.cast(inputs[:, :, 1, 0], dtype='int32')

        if self.arg == 'HZ':
            graph_in_1 = tf.gather(self.graph_fe_in_week_1, week)
            graph_in_2 = tf.gather(self.graph_fe_in_week_2, week)
        else:
            if self.arg == 'SH':
                graph_in_1 = tf.gather(self.graph_fe_in_date_1, date)
                graph_in_2 = tf.gather(self.graph_fe_in_date_2, date)
            else:
                graph_in_1 = tf.gather(self.graph_fe_in_date, date)
                graph_in_2 = tf.gather(self.graph_fe_in_week, week)

        delta_flow_4 = f - average[:, :4, :, :]

        # point-wise
        delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, date, graph_in_1, graph_in_2)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        #delta_flow_en = delta_flow_h
        #E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        #A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        #delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        #delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        # 没有激活函数

        delta_flow_de = self.filter_Out(delta_flow_en)

        ouputs = tf.nn.relu(delta_flow_de + average[:, 4:, :, :])

        return ouputs

class SGNN_test_gd(keras.Model):
    def __init__(self, graph_dim, stamps, days, mean, f_c=64, s_dim=16, l_num=2, pre_stamps=4, c=2,
                 total=False,
                 arg=None, name="SGNN", **kwargs):
        super(SGNN_test_gd, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.stamps = stamps
        self.days = days
        self.mean = mean
        self.f_c = f_c
        self.s_dim = s_dim
        self.l_num = l_num
        self.pre_stamps = pre_stamps
        self.c = c
        self.total = total
        self.arg = arg

        if self.arg == 'HZ':
            self.graph_fe_in_week_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
            self.graph_fe_in_week_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                      shape=[days, graph_dim, s_dim],
                                                      initializer=tf.keras.initializers.he_normal(seed))
        else:
            if self.arg == 'SH':
                self.graph_fe_in_date_1 = self.add_weight(name=self.name + '_graph_fe_in1',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_date_2 = self.add_weight(name=self.name + '_graph_fe_in2',
                                                          shape=[stamps, graph_dim, s_dim],
                                                          initializer=tf.keras.initializers.he_normal(seed))
            else:
                self.graph_fe_in_date = self.add_weight(name=self.name + '_graph_fe_in1',
                                                        shape=[stamps, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))
                self.graph_fe_in_week = self.add_weight(name=self.name + '_graph_fe_in2',
                                                        shape=[days, graph_dim, s_dim],
                                                        initializer=tf.keras.initializers.he_normal(seed))

        self.filter_In = tf.keras.layers.Conv2D(filters=f_c, kernel_size=1, name="x2embdingO" + "_" + self.name)
        self.mulit_linear_out = mulit_time_aware_layers(pre_stamps, graph_dim, stamps, l_num, f_c,
                                                        name=self.name + '_mulit_linear_out')

        self.E_s_Out = self.add_weight(name=self.name + '_E_sOut', shape=[graph_dim, graph_dim],
                                       initializer=tf.keras.initializers.he_normal(seed))

        self.gcn_thta = self.add_weight(name=self.name + '_thta', shape=[graph_dim, f_c, f_c],
                                        initializer=tf.keras.initializers.he_normal(seed))
        self.filter_Out = tf.keras.layers.Conv2D(filters=c, kernel_size=1, name="x2embding1" + "_" + self.name)

    def call(self, inputs):
        b = inputs.get_shape()[0]
        past = inputs[:, :, 2, 0]
        future = inputs[:, :, 2, 0] + 4
        date = tf.concat([past, future], axis=1)
        date = tf.cast(date, dtype='int32')

        if b != None:
            average = []
            if self.total:
                for i in range(b):
                    average.append(tf.gather(self.mean['total'], date[i]))
            else:
                for i in range(b):
                    if inputs[i, 0, 1, 0] >= 5.0:
                        average.append(tf.gather(self.mean['weekends'], date[i]))
                    else:
                        average.append(tf.gather(self.mean['weekdays'], date[i]))
            average = tf.stack(average, axis=0)
            average = tf.cast(average, dtype='float32')
        else:
            average = tf.concat([inputs[:, :, 3:, :], inputs[:, :, 3:, :]], axis=1)
        # average -> b, 8, 80, 2

        f = inputs[:, :, 3:, :]

        date = tf.cast(inputs[:, :, 2, 0], dtype='int32')
        week = tf.cast(inputs[:, :, 1, 0], dtype='int32')

        if self.arg == 'HZ':
            graph_in_1 = tf.gather(self.graph_fe_in_week_1, week)
            graph_in_2 = tf.gather(self.graph_fe_in_week_2, week)
        else:
            if self.arg == 'SH':
                graph_in_1 = tf.gather(self.graph_fe_in_date_1, date)
                graph_in_2 = tf.gather(self.graph_fe_in_date_2, date)
            else:
                graph_in_1 = tf.gather(self.graph_fe_in_date, date)
                graph_in_2 = tf.gather(self.graph_fe_in_week, week)

        delta_flow_4 = f - average[:, :4, :, :]

        # point-wise
        delta_flow_4 = self.filter_In(delta_flow_4)
        # x + time_embedding
        delta_flow_en = self.mulit_linear_out(delta_flow_4, date, graph_in_1, graph_in_2)

        # delta_flow_h = self.mulit_linear(delta_flow_h, date, x_embdding, graph_day_in, graph_week_in)

        #delta_flow_en = delta_flow_h
        E_h = tf.matmul(self.E_s_Out, self.E_s_Out, transpose_b=True)
        #
        A = tf.nn.softmax(tf.nn.relu(E_h), axis=1)
        delta_flow_g = tf.einsum('btge, gv->btve', delta_flow_en, A)
        # date = tf.cast(inputs[:, 0, 2, 0], dtype='int32')
        # thta = tf.gather(self.gcn_thta, date)
        delta_flow = tf.einsum('btge, gef->btgf', delta_flow_g, self.gcn_thta)
        # delta_flow = self.drop(delta_flow)
        # 没有激活函数

        delta_flow_de = self.filter_Out(delta_flow)

        ouputs = tf.nn.relu(delta_flow_de + average[:, 4:, :, :])

        return ouputs
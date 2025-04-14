from utils import *
import pickle
import numpy as np

# Dataset class
class Dataset(object):
    def __init__(self, x_data, average):
        self.__data = x_data
        self.average = average

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'weekends': self.average['weekends'], 'weekdays': self.average['weekdays'], 'total': self.average['total']}

    def get_len(self, type):
        return len(self.__data[type])


# Divide the data into train, val and test sequences
def data_gen(train_file, val_file, test_file, data_name='HZ', whole=False):
    f = open(train_file, 'rb')
    data = pickle.load(f)
    f.close()
    f = open(val_file, 'rb')
    val_data = pickle.load(f)
    f.close()
    f = open(test_file, 'rb')
    test_data = pickle.load(f)
    f.close()

    # train
    train_x_data = data['x']
    train_y_data = data['y']
    train_is_holiday = data['is_holiday']
    # 1188, 8, 80, 2
    data_train = np.concatenate([train_x_data, train_y_data], axis=1)

    val_x_data = val_data['x']
    val_y_data = val_data['y']
    val_is_holiday = val_data['is_holiday']
    # _, 8, 80, 2
    data_val = np.concatenate([val_x_data, val_y_data], axis=1)

    test_x_data = test_data['x']
    test_y_data = test_data['y']
    test_is_holiday = test_data['is_holiday']
    # _, 8, 80, 2
    data_test = np.concatenate([test_x_data, test_y_data], axis=1)

    ############################################################################
    if data_name == 'HZ':
        # 69, 80, 2
        h_s = np.zeros(shape=[73, 80, 2], dtype=np.float32)
        k_h = 0
        sum_f = np.zeros(shape=[73, 80, 2], dtype=np.float32)
        k_nh = 0

        # train #
        for i in range(17, 19):
            if train_is_holiday[i * 66] >= 5:
                h_s += np.concatenate(
                        [data_train[i * 66:(i + 1) * 66, 0, :, :], data_train[(i + 1) * 66 - 1, 1:, :, :]],
                        axis=0)
                k_h += 1
            else:
                sum_f += np.concatenate(
                        [data_train[i * 66:(i + 1) * 66, 0, :, :], data_train[(i + 1) * 66 - 1, 1:, :, :]],
                        axis=0)
                k_nh += 1
        # val #

        for i in range(2):
            if val_is_holiday[i * 66] >= 5:
                    h_s += np.concatenate(
                        [data_val[i * 66:(i + 1) * 66, 0, :, :], data_val[(i + 1) * 66 - 1, 1:, :, :]],
                        axis=0)
                    k_h += 1
            else:
                sum_f += np.concatenate(
                        [data_val[i * 66:(i + 1) * 66, 0, :, :], data_val[(i + 1) * 66 - 1, 1:, :, :]],
                        axis=0)
                k_nh += 1
        if whole:
            # test #
            for i in range(4):
                if test_is_holiday[i * 66] >= 5:
                    h_s += np.concatenate(
                            [data_test[i * 66:(i + 1) * 66, 0, :, :], data_test[(i + 1) * 66 - 1, 1:, :, :]],
                            axis=0)
                    k_h += 1
                else:
                    sum_f += np.concatenate(
                            [data_test[i * 66:(i + 1) * 66, 0, :, :], data_test[(i + 1) * 66 - 1, 1:, :, :]],
                            axis=0)
                    k_nh += 1

        weekends = h_s / k_h
        weekdays = sum_f / k_nh
        total = (h_s + sum_f) / (k_nh + k_h)
        average = {'weekends': weekends, 'weekdays': weekdays, 'total': total}
        start_time = '2019-01-01T05:30:00.000000000'

    if data_name == 'SH':
        h_s = np.zeros(shape=[73, 288, 2], dtype=np.float32)
        k_h = 0
        sum_f = np.zeros(shape=[73, 288, 2], dtype=np.float32)
        k_nh = 0
        # train #
        for i in range(62):
            if train_is_holiday[i * 66] >= 5:
                h_s += np.concatenate([data_train[i * 66:(i + 1) * 66, 0, :, :], data_train[(i + 1) * 66 - 1, 1:, :, :]],
                                    axis=0)
                k_h += 1
            else:
                sum_f += np.concatenate([data_train[i * 66:(i + 1) * 66, 0, :, :], data_train[(i + 1) * 66 - 1, 1:, :, :]],
                                        axis=0)
                k_nh += 1
        # val #

        for i in range(9):
            if val_is_holiday[i * 66] >= 5:
                h_s += np.concatenate([data_val[i * 66:(i + 1) * 66, 0, :, :], data_val[(i + 1) * 66 - 1, 1:, :, :]],
                                    axis=0)
                k_h += 1
            else:
                sum_f += np.concatenate([data_val[i * 66:(i + 1) * 66, 0, :, :], data_val[(i + 1) * 66 - 1, 1:, :, :]],
                                        axis=0)
                k_nh += 1
        if whole:
            # test #
            for i in range(21):
                if test_is_holiday[i * 66] >= 5:
                    h_s += np.concatenate(
                        [data_test[i * 66:(i + 1) * 66, 0, :, :], data_test[(i + 1) * 66 - 1, 1:, :, :]],
                        axis=0)
                    k_h += 1
                else:
                    sum_f += np.concatenate(
                        [data_test[i * 66:(i + 1) * 66, 0, :, :], data_test[(i + 1) * 66 - 1, 1:, :, :]],
                        axis=0)
                    k_nh += 1

        weekends = h_s / k_h
        weekdays = sum_f / k_nh
        total = (h_s + sum_f) / (k_nh + k_h)
        average = {'weekends': weekends, 'weekdays': weekdays, 'total': total}

        start_time = '2016-07-01T05:30:00.000000000'
    ############################################################################
    # 1188 -> 1188, 8, 1, 2
    xtime = data['xtime']
    ytime = data['ytime']
    train_time = np.concatenate([xtime, ytime], axis=1)
    train_time = train_time.astype(dtype='datetime64[60s]')
    train_time = train_time - np.datetime64(start_time, '60s')
    train_time = train_time.astype(dtype='float64')
    train_time = np.expand_dims(train_time, axis=[2, 3])
    train_time = np.tile(train_time, reps=[1, 1, 1, 2])

    #train_is_holiday = data['is_holiday']
    train_stamps = data['timestamps']

    # train_is_holiday = np.concatenate([train_is_holiday, train_is_holiday_arug], axis=0)
    train_is_holiday = np.expand_dims(train_is_holiday, axis=[1, 2, 3])
    train_is_holiday = np.tile(train_is_holiday, reps=[1, 8, 1, 2])

    # train_stamps = np.concatenate([train_stamps, train_stamps_arug], axis=0)
    train_stamps = np.expand_dims(train_stamps, axis=[1, 2, 3])
    train_stamps = np.tile(train_stamps, reps=[1, 1, 1, 2])
    tmp = [train_stamps]
    for i in range(1, 8):
        tmp.append(train_stamps + i)
    train_stamps = np.concatenate(tmp, axis=1)
    data_train = np.concatenate([train_time, train_is_holiday, train_stamps, data_train], axis=2)
    #############################################################################
    # val
    xtime = val_data['xtime']
    ytime = val_data['ytime']
    val_time = np.concatenate([xtime, ytime], axis=1)
    val_time = val_time.astype(dtype='datetime64[60s]')
    val_time = val_time - np.datetime64(start_time, '60s')
    val_time = val_time.astype(dtype='float64')
    val_time = np.expand_dims(val_time, axis=[2, 3])
    val_time = np.tile(val_time, reps=[1, 1, 1, 2])

    # _ -> _, 8, 1, 2
    #val_is_holiday = val_data['is_holiday']
    val_is_holiday = np.expand_dims(val_is_holiday, axis=[1, 2, 3])
    val_is_holiday = np.tile(val_is_holiday, reps=[1, 8, 1, 2])
    val_stamps = val_data['timestamps']
    val_stamps = np.expand_dims(val_stamps, axis=[1, 2, 3])
    val_stamps = np.tile(val_stamps, reps=[1, 1, 1, 2])
    tmp = [val_stamps]
    for i in range(1, 8):
        tmp.append(val_stamps + i)
    val_stamps = np.concatenate(tmp, axis=1)
    data_val = np.concatenate([val_time, val_is_holiday, val_stamps, data_val], axis=2)
    #############################################################################
    # test
    xtime = test_data['xtime']
    ytime = test_data['ytime']
    test_time = np.concatenate([xtime, ytime], axis=1)
    test_time = test_time.astype(dtype='datetime64[60s]')
    test_time = test_time - np.datetime64(start_time, '60s')
    test_time = test_time.astype(dtype='float64')
    test_time = np.expand_dims(test_time, axis=[2, 3])
    test_time = np.tile(test_time, reps=[1, 1, 1, 2])
    # _ -> _, 8, 1, 2
    #test_is_holiday = test_data['is_holiday']
    test_is_holiday = np.expand_dims(test_is_holiday, axis=[1, 2, 3])
    test_is_holiday = np.tile(test_is_holiday, reps=[1, 8, 1, 2])

    test_stamps = test_data['timestamps']
    test_stamps = np.expand_dims(test_stamps, axis=[1, 2, 3])
    test_stamps = np.tile(test_stamps, reps=[1, 1, 1, 2])
    tmp = [test_stamps]
    for i in range(1, 8):
        tmp.append(test_stamps + i)
    test_stamps = np.concatenate(tmp, axis=1)
    data_test = np.concatenate([test_time, test_is_holiday, test_stamps, data_test], axis=2)
    #############################################################################

    datas = {'train': data_train, 'val': data_val, 'test': data_test}
    dataset = Dataset(datas, average)
    return dataset


if __name__ == "__main__":
    # 两种方法都能打开

    # 5:15 - 23:30 15分钟 73
    # 31 * 0.6 = 18
    # 重新分配时间

    f = open('./dataset/hangzhou/train.pkl', 'rb')
    train_data = pickle.load(f)
    f.close()
    f = open('./dataset/hangzhou/val.pkl', 'rb')
    val_data = pickle.load(f)
    f.close()
    f = open('./dataset/hangzhou/test.pkl', 'rb')
    test_data = pickle.load(f)
    f.close()

    # 2019年1月1日、5日、6日、12日、13日、19日放假
    # 66 * 18
    # 一月1号是星期2
    # 5、6代表周末
    is_holiday = np.zeros(shape=66 * 19)
    timestamps = np.zeros(shape=66 * 19)
    for i in range(66 * 19):
        day = (int(i / 66) + 1) % 7
        is_holiday[i] = day
        #if day == 5 or day == 6:
         #  is_holiday[i] = 1
        stamps = i % 66
        timestamps[i] = stamps
    train_data['is_holiday'] = is_holiday
    train_data['timestamps'] = timestamps
    new_x = np.concatenate([train_data['x'], val_data['x'][:66, :, :, :]], axis=0)
    new_y = np.concatenate([train_data['y'], val_data['y'][:66, :, :, :]], axis=0)
    new_xtime = np.concatenate([train_data['xtime'], val_data['xtime'][:66, :]], axis=0)
    new_ytime = np.concatenate([train_data['ytime'], val_data['ytime'][:66, :]], axis=0)
    train_data['x'] = new_x
    train_data['y'] = new_y
    train_data['xtime'] = new_xtime
    train_data['ytime'] = new_ytime

    f = open('./dataset/hangzhou/new_train.pkl', 'wb')
    pickle.dump(train_data, f)
    f.close()

    # 20 是周末 21 工作日
    is_holiday = np.zeros(shape=66 * 2)
    timestamps = np.zeros(shape=66 * 2)
    for i in range(66 * 2):
        day = (int(i / 66) + 6) % 7
        is_holiday[i] = day
        #if day == 5 or day == 6:
         #  is_holiday[i] = 1
        stamps = i % 66
        timestamps[i] = stamps
    val_data['is_holiday'] = is_holiday
    val_data['timestamps'] = timestamps
    new_x = np.concatenate([val_data['x'][66:, :, :, :], test_data['x'][:66, :, :, :]], axis=0)
    new_y = np.concatenate([val_data['y'][66:, :, :, :], test_data['y'][:66, :, :, :]], axis=0)
    new_xtime = np.concatenate([val_data['xtime'][66:, :], test_data['xtime'][:66, :]], axis=0)
    new_ytime = np.concatenate([val_data['ytime'][66:, :], test_data['ytime'][:66, :]], axis=0)
    val_data['x'] = new_x
    val_data['y'] = new_y
    val_data['xtime'] = new_xtime
    val_data['ytime'] = new_ytime

    f = open('./dataset/hangzhou/new_val.pkl', 'wb')
    pickle.dump(val_data, f)
    f.close()


    # 26、27是周末
    # 22 23 24 25
    # 1 2 3 4
    is_holiday = np.zeros(shape=66 * 4)
    timestamps = np.zeros(shape=66 * 4)
    for i in range(66 * 4):
        day = (int(i / 66) + 1) % 7
        is_holiday[i] = day
        #if day == 5 or day == 6:
         #  is_holiday[i] = 1
        stamps = i % 66
        timestamps[i] = stamps
    test_data['is_holiday'] = is_holiday
    test_data['timestamps'] = timestamps

    new_x = test_data['x'][66:, :, :, :]
    new_y = test_data['y'][66:, :, :, :]
    new_xtime = test_data['xtime'][66:, :]
    new_ytime = test_data['ytime'][66:, :]

    test_data['x'] = new_x
    test_data['y'] = new_y
    test_data['xtime'] = new_xtime
    test_data['ytime'] = new_ytime

    f = open('./dataset/hangzhou/new_test.pkl', 'wb')
    pickle.dump(test_data, f)
    f.close()

    # img_path = './train_data.pkl'
    # img_data = np.load(img_path)
    # print(img_data)

    f = open('./dataset/shanghai/train.pkl', 'rb')
    train_data = pickle.load(f)
    # 4092, 4, 288, 2
    # 62 天
    #print(train_data['xtime'].shape)
    f.close()
    f = open('./dataset/shanghai/val.pkl', 'rb')
    val_data = pickle.load(f)
    # 9 天 594
    #print(val_data['xtime'].shape)
    f.close()
    f = open('./dataset/shanghai/test.pkl', 'rb')
    test_data = pickle.load(f)
    # 21 天 1386
    #print(test_data['xtime'].shape)
    f.close()

    # 2016年7月1日是星期5 没有公共假期
    # 66 * 62
    # 一月1号是星期2
    # 5、6代表周末
    is_holiday = np.zeros(shape=66 * 62)
    timestamps = np.zeros(shape=66 * 62)
    for i in range(66 * 62):
        day = (int(i / 66) + 4) % 7
        is_holiday[i] = day
        #if day == 5 or day == 6:
         #  is_holiday[i] = 1
        stamps = i % 66
        timestamps[i] = stamps
    train_data['is_holiday'] = is_holiday
    train_data['timestamps'] = timestamps

    f = open('./dataset/shanghai/new_train.pkl', 'wb')
    pickle.dump(train_data, f)
    f.close()

    # 2016 年 9 月 1 日是 星期4
    # 中秋：9月15日-17日放假，18日上班
    is_holiday = np.zeros(shape=66 * 9)
    timestamps = np.zeros(shape=66 * 9)
    for i in range(66 * 9):
        day = (int(i / 66) + 3) % 7
        is_holiday[i] = day
        #if day == 5 or day == 6:
         #  is_holiday[i] = 1
        stamps = i % 66
        timestamps[i] = stamps
    val_data['is_holiday'] = is_holiday
    val_data['timestamps'] = timestamps

    f = open('./dataset/shanghai/new_val.pkl', 'wb')
    pickle.dump(val_data, f)
    f.close()

    # 中秋：9月15日-17日放假，18日上班
    # 21 天
    # 10(星期6) 11 12 13 14(2->4) 15(3->5) 16(4->5) 17(5->6) 18(7->0) 19(0) 20
    # 0         1  2  3  4     5        6        7        8        9
    is_holiday = np.zeros(shape=66 * 21)
    timestamps = np.zeros(shape=66 * 21)
    for i in range(66 * 21):
        day = (int(i / 66) + 5) % 7
        if int(i / 66) == 4:
            day = 4
        if int(i / 66) == 5 or int(i / 66) == 6:
            day = 5
        if int(i / 66) == 7:
            day = 6
        if int(i / 66) == 8:
            day = 0
        is_holiday[i] = day
        #if day == 5 or day == 6:
         #  is_holiday[i] = 1
        stamps = i % 66
        timestamps[i] = stamps
    test_data['is_holiday'] = is_holiday
    test_data['timestamps'] = timestamps

    f = open('./dataset/shanghai/new_test.pkl', 'wb')
    pickle.dump(test_data, f)
    f.close()

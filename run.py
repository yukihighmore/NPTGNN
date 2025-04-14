import tensorflow as tf

tf.config.run_functions_eagerly(True)
from data_loader import *
from layer import *
from tensorflow import keras
from model import *

if __name__ == "__main__":

    Dataset_name = 'HZ'
    # _no_per
    # _no_dyna
    # _no_stamps_emb
    # _no_hd
    # _gnn
    # _test_gd
    if Dataset_name == 'SH':
        Data = data_gen('./dataset/shanghai/new_train.pkl', './dataset/shanghai/new_val.pkl',
                        './dataset/shanghai/new_test.pkl', data_name='SH')
        # 32 8 2
        model = SGNN(288, 69, 7, mean=Data.get_stats(), f_c=32, s_dim=32, l_num=4,
                     pre_stamps=4, c=2, total=False,
                     arg=Dataset_name)
        model.build(input_shape=(None, 4, 288 + 3, 2))
        early_patient = 25
        inital_lr = 0.001
        model_name = 'sh_best.keras'
    if Dataset_name == 'HZ':
        Data = data_gen('./dataset/hangzhou/new_train.pkl', './dataset/hangzhou/new_val.pkl',
                        './dataset/hangzhou/new_test.pkl', data_name='HZ')
        # 64 16 3
        model = SGNN(80, 69, 7, mean=Data.get_stats(), f_c=64, s_dim=16, l_num=3,
                     pre_stamps=4, c=2, total=True,
                     arg=Dataset_name)
        model.build(input_shape=(None, 4, 80 + 3, 2))
        early_patient = 50
        inital_lr = 0.002
        model_name = 'hz_best.keras'
    print('>> Loading dataset successfully!<<')

    # dataloader
    batch_size = 64
    train_data = Data.get_data('train')
    his_data = tf.constant(train_data[:, :4, :, :], dtype=tf.float32)
    pre_data = tf.constant(train_data[:, 4:, 3:, :], dtype=tf.float32)

    his_train_data = tf.data.Dataset.from_tensor_slices(his_data)
    pre_train_data = tf.data.Dataset.from_tensor_slices(pre_data)
    train_dataset = tf.data.Dataset.zip((his_train_data, pre_train_data)).shuffle(buffer_size=16).batch(
        batch_size).cache()

    val_data = Data.get_data('val')
    his_val_data = tf.data.Dataset.from_tensor_slices(tf.constant(val_data[:, :4, :, :], dtype=tf.float32))
    pre_val_data = tf.data.Dataset.from_tensor_slices(tf.constant(val_data[:, 4:, 3:, :], dtype=tf.float32))
    val_dataset = tf.data.Dataset.zip((his_val_data, pre_val_data)).shuffle(buffer_size=16).batch(batch_size).cache()
    test_data = Data.get_data('test')
    his_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(test_data[:, :4, :, :], dtype=tf.float32))
    pre_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(test_data[:, 4:, 3:, :], dtype=tf.float32))
    test_dataset = tf.data.Dataset.zip((his_test_data, pre_test_data)).shuffle(buffer_size=16).batch(batch_size).cache()

    epochs = 300

    checkpoint_save_path = 'model/' + model_name + '.ckpt'

    # period=1 verbose=1
    optimizer = tf.keras.optimizers.Adam(learning_rate=inital_lr)
    import keras.backend as K
    from keras.callbacks import LearningRateScheduler

    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/2
        if epoch % 30 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    callbacks = [
        reduce_lr,
        keras.callbacks.ModelCheckpoint(monitor='val_mean_absolute_error', mode='min',
                                        save_best_only=True,
                                        save_weights_only=True,
                                        filepath=checkpoint_save_path),
        #keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=10,
        #                                 mode='min', cooldown=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.0001,
                                      patience=early_patient)
    ]

    # tf.keras.losses.mean_absolute_error
    loss = tf.keras.losses.Huber(delta=2.0, name="mean_absolute_error")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=tf.keras.metrics.MeanAbsoluteError(), run_eagerly=True)
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    # test
    model.load_weights(checkpoint_save_path)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_save_path).expect_partial()

    y_pred = []
    y_true = []
    for element in test_dataset.as_numpy_iterator():
        x, y = element
        #y_pred_inflow, y_pred_outflow = outflow_model.predict(x)
        y_ = model.predict(x)
        #y_pred_outflow =model.predict(x)
        #y_ = tf.stack([y_pred_inflow, y_pred_outflow], axis=-1)
        y_true.append(y)
        y_pred.append(y_)
    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)
    evl = evaluation(y_true, y_pred)['inflow']
    print('****************inflow*******************')
    mape = 0
    mae = 0
    rmse = 0
    for step in range(4):
        # print(f'{evl[step][0]:7.3%},{evl[step][1]:4.3f},{evl[step][2]:6.3f}')
        print(
            f'The steps {step} MAPE is: {evl[step][0]:7.3%}, MAE is:{evl[step][1]:4.3f}, RMSE is: {evl[step][2]:6.3f}')
        mape = mape + evl[step][0]
        mae = mae + evl[step][1]
        rmse = rmse + evl[step][2]
    mape = mape / 4
    mae = mae / 4
    rmse = rmse / 4
    # print(f'{mape:7.3%},{mae:4.3f},{rmse:6.3f}')
    print(f'The average MAPE is: {mape:7.3%}, MAE is: {mae:4.3f}, RMSE is: {rmse:6.3f}')
    print('****************outflow*******************')
    evl = evaluation(y_true, y_pred)['outflow']
    mape = 0
    mae = 0
    rmse = 0
    for step in range(4):
        # print(f'{evl[step][0]:7.3%},{evl[step][1]:4.3f},{evl[step][2]:6.3f}')
        print(
            f'The steps {step} MAPE is: {evl[step][0]:7.3%}, MAE is:{evl[step][1]:4.3f}, RMSE is: {evl[step][2]:6.3f}')
        mape = mape + evl[step][0]
        mae = mae + evl[step][1]
        rmse = rmse + evl[step][2]
    mape = mape / 4
    mae = mae / 4
    rmse = rmse / 4
    # print(f'{mape:7.3%},{mae:4.3f},{rmse:6.3f}')
    print(f'The average MAPE is: {mape:7.3%}, MAE is: {mae:4.3f}, RMSE is: {rmse:6.3f}')
    print('****************average*******************')
    evl = evaluation(y_true, y_pred)['average']
    mape = 0
    mae = 0
    rmse = 0
    for step in range(4):
        # print(f'{evl[step][0]:7.3%},{evl[step][1]:4.3f},{evl[step][2]:6.3f}')
        print(
            f'The steps {step} MAPE is: {evl[step][0]:7.3%}, MAE is:{evl[step][1]:4.3f}, RMSE is: {evl[step][2]:6.3f}')
        mape = mape + evl[step][0]
        mae = mae + evl[step][1]
        rmse = rmse + evl[step][2]
    mape = mape / 4
    mae = mae / 4
    rmse = rmse / 4
    # print(f'{mape:7.3%},{mae:4.3f},{rmse:6.3f}')
    print(f'The average MAPE is: {mape:7.3%}, MAE is: {mae:4.3f}, RMSE is: {rmse:6.3f}')

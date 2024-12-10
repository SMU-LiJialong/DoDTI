#!/usr/bin/env python
# encoding: utf-8

#****************************
# training
#***************************

import os
import config
import h5py
import numpy as np
import tensorflow as tf
import functions_wlls_3D as func
import net_dodti as net
from tensorflow.keras import mixed_precision


#%% learning rate
def lr_schedule(epoch):
    if epoch <= 100:
        lr = 0.0001
    elif epoch <= 200:
        lr = 0.00005
    else:
        lr = 0.00001
    return lr


#%% display the learning rate in console
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))


#%% Loss
class LossMaeMaskWt_mask(tf.keras.losses.Loss):  # DW to DW, mask
    def call(self, y_true, y_pred):
        # y_pred   = y_pred[-1]                  # Calculate the loss using only the final output
        num_stage = tf.cast(tf.shape(y_pred)[0] - 1, tf.float32) / 3
        ws = tf.repeat(tf.range(1, num_stage + 2, dtype=tf.float32), 3)
        ws = ws[:-2]
        w = tf.cast(tf.math.divide(ws, tf.math.reduce_sum(ws)), dtype=y_pred.dtype.name)

        true_s0, true_d, mask = tf.split(y_true, [1, 6, 1], axis=-1)
        new_true = tf.concat([true_s0, true_d * 1000.0], axis=-1)
        mask = tf.broadcast_to(mask, y_pred.get_shape().as_list())
        new_true = tf.broadcast_to(new_true, y_pred.get_shape().as_list()) * mask
        new_pred = y_pred * mask

        err = tf.math.abs(new_true - new_pred)
        loss = tf.math.reduce_mean(err, axis=[1, 2, 3, 4, 5])
        loss = tf.math.reduce_sum(loss * w)
        return loss


def preproc_data(path_train, path_valid):
    path_bvecs = './data/gradient/bvecs_opt.txt'
    path_bvals = './data/gradient/bvals_opt.txt'
    bvecs = np.loadtxt(path_bvecs, dtype='float32')
    bvals = np.loadtxt(path_bvals, dtype='float32')
    bvals = bvals / 1000.0

    f = h5py.File(path_train, 'r')
    train_nf = np.array(f['noise free data'], dtype='float32')
    train_n = np.array(f['noise data'], dtype='float32')
    train_s0 = np.array(f['s0 map'], dtype='float32')
    train_d = np.array(f['d map'], dtype='float32')  # diffusion tensor filed: Dxx Dyy Dzz Dxy Dyz Dxz
    train_mask = np.array(f['mask'], dtype='float32')
    f.close()

    f = h5py.File(path_valid, 'r')
    valid_nf = np.array(f['noise free data'], dtype='float32')
    valid_n = np.array(f['noise data'], dtype='float32')
    valid_s0 = np.array(f['s0 map'], dtype='float32')
    valid_d = np.array(f['d map'], dtype='float32')
    valid_mask = np.array(f['mask'], dtype='float32')
    f.close()

    bvals_train = np.broadcast_to(bvals, (train_nf.shape[0], bvals.shape[0]))
    bvecs_train = np.broadcast_to(bvecs, (train_nf.shape[0], bvecs.shape[0], bvecs.shape[1]))
    bvals_valid = np.broadcast_to(bvals, (valid_nf.shape[0], bvals.shape[0]))
    bvecs_valid = np.broadcast_to(bvecs, (valid_nf.shape[0], bvecs.shape[0], bvecs.shape[1]))

    #%% DW to map
    train_s0 = np.maximum(train_s0, 1e-8)
    valid_s0 = np.maximum(valid_s0, 1e-8)
    train_n = np.maximum(train_n, 1e-8)
    valid_n = np.maximum(valid_n, 1e-8)
    train_lns0 = -1.0 * np.log(train_s0)
    valid_lns0 = -1.0 * np.log(valid_s0)
    train_n = -1.0 * np.log(train_n)
    valid_n = -1.0 * np.log(valid_n)

    train_x = (train_n, bvecs_train, bvals_train, train_mask)
    train_y = np.concatenate([train_lns0, train_d, train_mask], axis=-1)
    valid_x = (valid_n, bvecs_valid, bvals_valid, valid_mask)
    valid_y = np.concatenate([valid_lns0, valid_d, valid_mask], axis=-1)
    return train_x, train_y, valid_x, valid_y


#%%
if __name__ == "__main__":
    print('START')
    # config.config_gpu(0)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size = 4           # batch size
    epochs = 501             # epochs
    data_name = './exp1_mix' # experiment name
    num_model = 1            # model number
    save_every = 5           # save the model every how many epochs
    Ns = 8                   # number of stage for ADMM
    Nf = 1                   # number of iteration in fitting block
    Nc = 7                   # input image channel(the number of diffusion encoding directions)
    recon_flag = False       # whether to reconstruct the DW image from the diffusion tensor field in the output
    bn_flag = False          # whether to use batch normalization (BN) layers
    mix_precision = False    # whether to use mixed precision
    denoiser = 'DnCNN'       # denoiser use DnCNN or ResNet
    Nblock = 7               # number of convolutional layers
    Nfilters = 64            # number of convolutional kernels
    kernel_size = 3          # convolutional kernel size

    #%% 
    if mix_precision is True:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    #%% DATASET
    # Train and validation dataset
    # path_train = '/public2/lijialong/Project/unrollingnet/ADMM_LJL/exp8_mix/data_train/simulated_train_blocks_0_32.h5'
    # path_valid = '/public2/lijialong/Project/unrollingnet/ADMM_LJL/exp8_mix/data_valid/simulated_valid_blocks_0_32.h5'
    path_train = os.path.join(data_name, 'data_train', 'simulated_train_blocks_demo.h5')
    path_valid = os.path.join(data_name, 'data_valid', 'simulated_valid_blocks_demo.h5')
    train_x, train_y, valid_x, valid_y = preproc_data(path_train, path_valid)
    num_train = train_y.shape[0]
    num_valid = valid_y.shape[0]

    steps_per_epoch = int(num_train / batch_size)
    print(steps_per_epoch)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).cache().shuffle(buffer_size=num_train). \
        batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).cache().shuffle(buffer_size=num_valid). \
        batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #%% Model configuration
    model_dir = os.path.join(data_name, 'model', '3D_WLLS_{}_{}'.format(denoiser, num_model))
    admmnet = net.ADMMNetm(Ns=Ns, Nf=Nf, Na=1, Nblock=Nblock, Nfilters=Nfilters, f=kernel_size,
                           denoiser=denoiser, recon_flag=recon_flag, bn_flag=bn_flag, name='WLLS_' + denoiser)
    inpt_b = tf.keras.layers.Input(shape=(None, None, None, Nc), name='input_dti')
    inpt_bvecs = tf.keras.layers.Input(shape=(Nc, 3), name='bvecs')
    inpt_bvals = tf.keras.layers.Input(shape=(Nc,), name='bvals')
    inpt_mask = tf.keras.layers.Input(shape=(None, None, None, 1), name='mask')
    para_map = admmnet([inpt_b, inpt_bvecs, inpt_bvals, inpt_mask])
    model = tf.keras.Model(inputs=[inpt_b, inpt_bvecs, inpt_bvals, inpt_mask], outputs=para_map)
    model.summary()

    opti = tf.keras.optimizers.Adam(learning_rate=0.0001)
    if mix_precision is True:
        opti = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer=opti, loss_scale='dynamic')
    loss = LossMaeMaskWt_mask()
    model.compile(optimizer=opti, loss=loss)

    initial_epoch = func.findLastCheckPoint(save_dir=model_dir)
    if initial_epoch > 0:
        print('Resuming by loading epoch %04d' % initial_epoch)
        model.load_weights(filepath=os.path.join(model_dir, 'model_%04d.h5' % initial_epoch))

    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'model_{epoch:04d}.h5'), verbose=1,
                                                      save_weights_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir, 'log.csv'), append=True, separator=',')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1, profile_batch=0)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              initial_epoch=initial_epoch,
              callbacks=[checkpointer, csv_logger, tensorboard, lr_scheduler, PrintLR()])

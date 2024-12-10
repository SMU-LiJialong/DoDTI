#!usr/bin/env python
# encoding: utf-8

#****************************
# Package the noisy data, noise-free data, diffusion tensor, and mask into training and validation datasets
#***************************

import os
import h5py
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

#%%
def create_folder(data_name):
    if not os.path.exists(data_name):
        os.mkdir(data_name)

    path_train = os.path.join(data_name, 'data_train')
    if not os.path.exists(path_train):
        os.mkdir(path_train)

    path_valid = os.path.join(data_name, 'data_valid')
    if not os.path.exists(path_valid):
        os.mkdir(path_valid)
    print('\n>>>Finish: make folders\n')


def data_aug(img, mode=0):
    # aug data size
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


#%% main function ***************************************************************
if __name__ == '__main__':
    # parameter setting
    data_name = 'exp1_mix'  # Experiment number, mixed noise
    # ratio_train_val = 16/20 # train/(train+valid)
    ratio_train_val = 1/2   # this demo only include one subject; set as 1/2 for running
    scales = [1]
    sz_patch = 32           # patch size
    stride = 16             # slide step size for extracting patch
    aug_times = 1           # number of data augment：
    rate_discard = 0.1      # Ratio of discarded background voxels
    create_folder(data_name)

    # for id_data in range(1, 21):
        # num = '%02d' % id_data
    for id_data in range(1, 3): # for demo, both the training and validation datasets use the same data
        num = '01'  

        # load data
        noise_free_path = r'./data/nf_norm/nf_norm_'+num+'.nii.gz'
        noise_path = r'./data/rician_norm/rician_'+num+'.nii.gz'
        tensor_path = r'./data/tensor/tensor_cwlls_'+num+'.nii.gz'
        mask_path = r'./data/mask/mask_'+num+'.nii.gz'
        print(noise_free_path)
        print(noise_path)
        print(tensor_path)
        print(mask_path)

        num_nf = nib.load(noise_free_path).get_fdata().astype('float32')
        num_s0 = num_nf[..., 0:1]
        num_n = nib.load(noise_path).get_fdata().astype('float32')
        num_tensor = nib.load(tensor_path).get_fdata().astype('float32')
        num_mask = nib.load(mask_path).get_fdata().astype('float32')
        num_mask = num_mask[..., np.newaxis]
        print('max(noise_free)=', np.max(num_n))
        print('max(tensor)=', np.max(num_tensor))
        print('\n')

        # Eliminate marginal useless voxels
        k = 6
        dim1, dim2, dim3, dirs = num_nf.shape
        num_nf = num_nf[np.newaxis, k:dim1-k, k:dim2-k, :, :]   # 140--->128, suitable for extract patches
        num_s0 = num_s0[np.newaxis, k:dim1-k, k:dim2-k, :, :]
        num_n = num_n[np.newaxis, k:dim1-k, k:dim2-k, :, :]
        num_tensor = num_tensor[np.newaxis, k:dim1-k, k:dim2-k, :, :]
        num_mask = num_mask[np.newaxis, k:dim1-k, k:dim2-k, :, :]

        # Verify that dti is consistent with the mask
        # plt.figure(figsize=(4, 2), dpi=200), plt.axis('off')
        # plt.imshow(np.hstack([num_nf[50,:,:,1], num_n[50,:,:,1], num_mask[50,:,:,0]]), cmap='gray')
        # plt.savefig('./check/check1.png')
        # plt.show()

        # append all data
        if id_data == 1:
            data_all_nf = num_nf
            data_all_s0 = num_s0
            data_all_n = num_n
            data_all_tensor = num_tensor
            data_all_mask = num_mask
        else:
            data_all_nf = np.append(data_all_nf, num_nf, axis=0)
            data_all_s0 = np.append(data_all_s0, num_s0, axis=0)
            data_all_n = np.append(data_all_n, num_n, axis=0)
            data_all_tensor = np.append(data_all_tensor, num_tensor, axis=0)
            data_all_mask = np.append(data_all_mask, num_mask, axis=0)

    data_all_nf = np.array(data_all_nf)
    data_all_n = np.array(data_all_n)
    data_all_s0 = np.array(data_all_s0)
    data_all_tensor = np.array(data_all_tensor)
    data_all_mask = np.array(data_all_mask)

    print('noise_free shape: ', data_all_nf.shape)
    print('noise shape:      ', data_all_n.shape)
    print('s0 map shape:     ', data_all_s0.shape)
    print('tensor shape:     ', data_all_tensor.shape)
    print('mask shape:       ', data_all_mask.shape)

    #%% extract training patches and data augmentation
    data_train_patches_nf = []
    data_train_patches_n = []
    data_train_patches_s0 = []
    data_train_patches_d = []
    data_train_patches_mask = []
    count_train_useless = 0
    num_all, h, w, l, c = data_all_nf.shape
    for id_study in range(int(num_all*ratio_train_val)):  # training data part
        for scale in scales:
            h_scaled, w_scaled, l_scaled, = int(h * scale), int(w * scale), int(l * scale)
            # extract 3D patches
            for i in range(0, h_scaled - sz_patch + 1, stride):
                for j in range(0, w_scaled - sz_patch + 1, stride):
                    for k in range(0, l_scaled - sz_patch + 1, stride):
                        xmk = data_all_mask[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                        if np.sum(xmk) > np.size(xmk) * rate_discard:
                            xnf = data_all_nf[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                            xn = data_all_n[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                            xs0 = data_all_s0[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                            xd = data_all_tensor[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]

                            # data augumentation
                            for k in range(0, aug_times):
                                # mode = np.random.randint(0, 8)
                                mode = 0
                                xnf_aug = data_aug(xnf, mode=mode)
                                xn_aug  = data_aug(xn, mode=mode)
                                xs0_aug = data_aug(xs0, mode=mode)
                                xd_aug  = data_aug(xd, mode=mode)
                                xmk_aug = data_aug(xmk, mode=mode)
    
                                data_train_patches_nf.append(xnf_aug)
                                data_train_patches_n.append(xn_aug)
                                data_train_patches_s0.append(xs0_aug)
                                data_train_patches_d.append(xd_aug)
                                data_train_patches_mask.append(xmk_aug)
                        else:
                            count_train_useless += 1

    data_train_patches_nf = np.array(data_train_patches_nf, dtype='float32')
    data_train_patches_n = np.array(data_train_patches_n, dtype='float32')
    data_train_patches_s0 = np.array(data_train_patches_s0, dtype='float32')
    data_train_patches_d = np.array(data_train_patches_d, dtype='float32')
    data_train_patches_mask = np.array(data_train_patches_mask, dtype='float32')

    # random permutation
    count_useless = count_train_useless * aug_times   # data argumentation
    count_train_patch = data_train_patches_nf.shape[0]
    index_pixel = np.array(range(0, count_train_patch))
    index_pixel_random = np.random.permutation(index_pixel)
    
    data_train_patches_nf = data_train_patches_nf[index_pixel_random]
    data_train_patches_n = data_train_patches_n[index_pixel_random]
    data_train_patches_s0 = data_train_patches_s0[index_pixel_random]
    data_train_patches_d = data_train_patches_d[index_pixel_random]
    data_train_patches_mask = data_train_patches_mask[index_pixel_random]
    
    print('train noise_free shape: ', data_train_patches_nf.shape)
    print('train noise shape:      ', data_train_patches_n.shape)
    print('train s0 map shape:     ', data_train_patches_s0.shape)
    print('train tensor shape:     ', data_train_patches_d.shape)
    print('train mask shape:       ', data_train_patches_mask.shape)

    #%% extract validation patches and data augmentation
    data_valid_patches_nf = []
    data_valid_patches_n = []
    data_valid_patches_s0 = []
    data_valid_patches_d = []
    data_valid_patches_mask = []
    count_valid_useless = 0
    num_all, h, w, l, c = data_all_nf.shape
    for id_study in range(int(num_all*ratio_train_val), num_all):  # validation data part
        for scale in scales:
            h_scaled, w_scaled, l_scaled, = int(h * scale), int(w * scale), int(l * scale)
            # extract 3D patches
            for i in range(0, h_scaled - sz_patch + 1, stride):
                for j in range(0, w_scaled - sz_patch + 1, stride):
                    for k in range(0, l_scaled - sz_patch + 1, stride):
                        xmk = data_all_mask[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                        if np.sum(xmk) > np.size(xmk) * rate_discard:
                            xnf = data_all_nf[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                            xn = data_all_n[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                            xs0 = data_all_s0[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]
                            xd = data_all_tensor[id_study, i:i + sz_patch, j:j + sz_patch, k:k + sz_patch, :]

                            # data augumentation
                            for k in range(0, aug_times):
                                # mode = np.random.randint(0, 8)
                                mode = 0
                                xnf_aug = data_aug(xnf, mode=mode)
                                xn_aug  = data_aug(xn, mode=mode)
                                xs0_aug = data_aug(xs0, mode=mode)
                                xd_aug  = data_aug(xd, mode=mode)
                                xmk_aug = data_aug(xmk, mode=mode)

                                data_valid_patches_nf.append(xnf_aug)
                                data_valid_patches_n.append(xn_aug)
                                data_valid_patches_s0.append(xs0_aug)
                                data_valid_patches_d.append(xd_aug)
                                data_valid_patches_mask.append(xmk_aug)
                        else:
                            count_valid_useless += 1

    data_valid_patches_nf = np.array(data_valid_patches_nf, dtype='float32')
    data_valid_patches_n = np.array(data_valid_patches_n, dtype='float32')
    data_valid_patches_s0 = np.array(data_valid_patches_s0, dtype='float32')
    data_valid_patches_d = np.array(data_valid_patches_d, dtype='float32')
    data_valid_patches_mask = np.array(data_valid_patches_mask, dtype='float32')
    
    # random permutation
    count_useless = count_valid_useless * aug_times   # data argumentation
    count_valid_patch = data_valid_patches_nf.shape[0]
    index_pixel = np.array(range(0, count_valid_patch))
    index_pixel_random = np.random.permutation(index_pixel)
    
    data_valid_patches_nf = data_valid_patches_nf[index_pixel_random]
    data_valid_patches_n = data_valid_patches_n[index_pixel_random]
    data_valid_patches_s0 = data_valid_patches_s0[index_pixel_random]
    data_valid_patches_d = data_valid_patches_d[index_pixel_random]
    data_valid_patches_mask = data_valid_patches_mask[index_pixel_random]
    
    print('valid noise_free shape: ', data_valid_patches_nf.shape)
    print('valid noise shape:      ', data_valid_patches_n.shape)
    print('valid s0 map shape:     ', data_valid_patches_s0.shape)
    print('valid tensor shape:     ', data_valid_patches_d.shape)
    print('valid mask shape:       ', data_valid_patches_mask.shape)
    
    # %%save train data 
    f = h5py.File(os.path.join(data_name, 'data_train', 'simulated_train_blocks_demo.h5'), 'w')
    f['noise free data'] = np.array(data_train_patches_nf)
    f['noise data']      = np.array(data_train_patches_n)
    f['s0 map']          = np.array(data_train_patches_s0)
    f['d map']           = np.array(data_train_patches_d)
    f['mask']            = np.array(data_train_patches_mask)
    f.close()

    # save validation data
    f = h5py.File(os.path.join(data_name, 'data_valid', 'simulated_valid_blocks_demo.h5'), 'w')
    f['noise free data'] = np.array(data_valid_patches_nf)
    f['noise data']      = np.array(data_valid_patches_n)
    f['s0 map']          = np.array(data_valid_patches_s0)
    f['d map']           = np.array(data_valid_patches_d)
    f['mask']            = np.array(data_valid_patches_mask)
    f.close()

    print('>> Training blocks:\t ' + str(data_train_patches_n.shape[0]) + ' (Useful=' + str(count_train_patch) + ', Useless=' + str(count_train_useless) + ')')
    print('>> Validation blocks:\t ' + str(data_valid_patches_n.shape[0]) + ' (Useful=' + str(count_valid_patch) + ', Useless=' + str(count_valid_useless) + ')')
    print('Finish data generation')

    # %% check
    i = 10
    slice = 24
    plt.figure(figsize=(2, 10), dpi=300)
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    plt.subplot(511), plt.axis('off')
    plt.imshow(data_train_patches_nf[i, :, :, slice, 0], cmap='gray', vmin=0, vmax=0.5)
    plt.subplot(512), plt.axis('off')
    plt.imshow(data_train_patches_n[i, :, :, slice, 0], cmap='gray', vmin=0, vmax=0.5)
    plt.subplot(513), plt.axis('off')
    plt.imshow(data_train_patches_s0[i, :, :, slice, 0], cmap='gray', vmin=0, vmax=0.5)
    plt.subplot(514), plt.axis('off')
    plt.imshow(data_train_patches_d[i, :, :, slice, 0], cmap='gray', vmin=0, vmax=0.003)
    plt.subplot(515), plt.axis('off')
    plt.imshow(data_train_patches_mask[i, :, :, slice, 0], cmap='gray', vmin=0, vmax=1)
    plt.savefig('./check/data_combined.png')
    plt.show()
    plt.close()


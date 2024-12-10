#!usr/bin/env python
# encoding: utf-8

#****************************
# Add noise for simulated training data
# In our paper, the training data consists of 20 subjects.
# However, this demo only includes data for subject01.
# Therefore, you will need to adjust this code accordingly.
#***************************

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

noise_type = 'rician'
for i in range(1, 21):   
    # load data
    num = '%02d' % i
    load_path = './data/nf_unnorm/recon_b1000_6dir_'+num+'.nii.gz'  # the simulated noise-free data (unnormazlied) for training
    save_path_n = './data/rician_norm/rician_'+num+'.nii.gz'        # save as simulated noisy data (normazlied)
    save_path_nf = './data/nf_norm/nf_norm_'+num+'.nii.gz'          # save as simulated noise-free data (normazlied)
    bvals = np.loadtxt("./data/gradient/bvals_opt.txt")
    img = nib.load(load_path)
    img_affine = img.affine
    data_nf = img.get_fdata().astype('float32')

    #%% normalization to noise free, excluded outliers
    index_b0 = np.where(bvals == 0)
    data_nf_flatten = np.sort(data_nf[..., index_b0].flatten())
    index_max = round(len(data_nf_flatten) * 0.999)  # Excluded outliers
    pseudo_max = data_nf_flatten[index_max]
    data_nf_norm = (data_nf-np.min(data_nf))/(pseudo_max-np.min(data_nf))
    print('nf_max:%f, psudo_max:%f, min:%f' % (np.max(data_nf), pseudo_max, np.min(data_nf)))
    print('mean: ', np.mean(data_nf))

    #%% noise level for each subject
    noise_level_train = np.linspace(0.005, 0.045, 16)  # 16 subjects for training
    noise_level_valid = np.linspace(0.01, 0.04, 4)     # 4 subjects for validation
    noise_level = np.concatenate([noise_level_train, noise_level_valid])
    # print(noise_level)
    sigma = noise_level[i-1]

    #%% adding noise according to noise type and sigma_p
    nr = sigma * np.random.standard_normal(data_nf_norm.shape) + data_nf_norm
    if noise_type == 'guassian':
        data_n_norm = nr
        print('Adding guassian noise, sigma=', sigma)
    elif noise_type == 'rician':
        ni = sigma * np.random.standard_normal(data_nf_norm.shape)
        data_n_norm = np.sqrt(nr ** 2 + ni ** 2)
        print('Adding rician noise, sigma=', sigma)
    else:
        print('noise type error')

    #%%  show nf and n for checking visiual effects of noise
    plt.figure(figsize=(4, 4))
    plt.subplots_adjust(left=0, right=1, wspace=0, hspace=0, bottom=0, top=1)
    plt.subplot(211), plt.title(sigma)
    plt.imshow(np.hstack((data_nf_norm[56,:,:,0], data_n_norm[56,:,:,0])), vmin=0, vmax=0.5, cmap='gray'), plt.axis('off')
    plt.subplot(212), plt.title(pseudo_max)
    plt.imshow(np.hstack([data_nf_norm[56,:,:,1], data_n_norm[56,:,:,1]]), vmin=0, vmax=0.3, cmap='gray'), plt.axis('off')
    plt.savefig('./check/add_train_noise.png')
    plt.show()
    plt.close()

    #%% save data_n as nii.gz
    nib.Nifti1Image(data_nf_norm, img_affine).to_filename(save_path_nf)
    nib.Nifti1Image(data_n_norm, img_affine).to_filename(save_path_n)
    print('Finish adding noise to data_%s with sigma=%f \n' % (num, sigma))
print('ok')

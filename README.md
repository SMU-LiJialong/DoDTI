# DoDTI
The implement of the following paper: "Reliable Deep Diffusion Tensor Estimation: Rethinking the Power of Data-Driven Optimization Routine". 

## Architecture of the proposed DoDTI
![image](https://github.com/user-attachments/assets/9832aa5e-a4db-477c-a04c-6c8d0c05b7ef)
Architecture of the proposed DoDTI. (a) The input includes at least six diffusion-encoding volumes plus one non-DW volume, and corresponding gradient directions and b values. The output includes the predicted non-DW volume and six tensor element maps. The n-th iteration of ADMM corresponds to the n-th stage of the deep neural network enclosed within the dashed box, in which the fitting block (X), the auxiliary variable block (Z), and the multiplier block (M) are sequentially updated. (b) The structure of the denoiser, which is embedded within the auxiliary variable block.

## data_negerateor_3D_mix.py
Generate paired data sets for training and validation using noise-free DTI data, noisy DTI data, and diffusion tensor field.

## train_RED_WLLS_3D.py
Training DoDTI.

## test_exp1_clean.py
Testing DoDTI.

## HCP data
The example data are provided by the Human Connectome Project. Please acknowledge the source to the WU-Minn-Oxford HCP. The orginal data is available at [https://www.humanconnectome.org/study/hcp-young-adult](https://www.humanconnectome.org/study/hcp-young-adult/).

## Refereces
[1] Li, J., Zhang, Z., Chen, Y., Lu, Q., Wu, Y., Liu, X., Feng, Q., Feng, Y. and Zhang, X., 2024. Reliable Deep Diffusion Tensor Estimation: Rethinking the Power of Data-Driven Optimization Routine. arXiv preprint arXiv:2409.02492.

[2] Fan, Q., Witzel, T., Nummenmaa, A., Van Dijk, K.R.A., Van Horn, J.D., Drews, M.K., Somerville, L.H., Sheridan, M.A., Santillana, R.M., Snyder, J., Hedden, T., Shaw, E.E., Hollinshead, M.O., Renvall, V., Zanzonico, R., Keil, B., Cauley, S., Polimeni, J.R., Tisdall, D., Buckner, R.L., Wedeen, V.J., Wald, L.L., Toga, A.W., Rosen, B.R., 2016. MGH-USC Human Connectome Project datasets with ultra-high b-value diffusion MRI. Neuroimage 124, 1108-1114.

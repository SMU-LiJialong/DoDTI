#!/usr/bin/env python
# encoding: utf-8

#****************************
# the dodti network
#***************************

import tensorflow as tf
import functions_wlls_3D as func
import net_denoiser

class FittingLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(FittingLayer, self).__init__(name=name)

    def call(self, x, z, beta, rho, sigma, y, bvecs, bvals, signal, mask):
        # %% Method 1: Use matrix inversion (inv) or apply Newton method
        # rho = tf.cast(rho, dtype=x.dtype.name)
        # g1 = bvecs[..., 0]  # [N,1,1,Nb]
        # g2 = bvecs[..., 1]
        # g3 = bvecs[..., 2]

        # dSds0 = tf.ones_like(g1)  # [N,1,1,Nb]
        # dSddxx = bvals * g1 * g1
        # dSddyy = bvals * g2 * g2
        # dSddzz = bvals * g3 * g3
        # dSddxy = 2.0 * bvals * g1 * g2
        # dSddxz = 2.0 * bvals * g1 * g3
        # dSddyz = 2.0 * bvals * g2 * g3
        # a = tf.stack([dSds0, dSddxx, dSddyy, dSddzz, dSddxy, dSddyz, dSddxz], axis=-1)  # [N,1,1,Nb,7]
        # I_matrix = tf.eye(7)
        # w = signal*signal    # [N,Nx,Ny,Nb]  weighting factor
        # fir = tf.linalg.inv(tf.matmul(a, w[..., tf.newaxis]*a, transpose_a=True) + rho * I_matrix[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :])  # iv=inv(aTwa+ρI)
        # sec = tf.matmul(a, w[..., tf.newaxis] * y[..., tf.newaxis], transpose_a=True) + tf.expand_dims(rho * (z - beta), -1)
        # x = tf.squeeze(tf.matmul(fir, sec), -1)

        # %% Method 2: newton
        rho = tf.cast(rho, dtype=x.dtype.name)
        gx = bvecs[..., 0]  # [N,1,1,Nb]
        gy = bvecs[..., 1]
        gz = bvecs[..., 2]
        w = tf.math.divide_no_nan(1.0, tf.math.square(sigma))
        w = tf.cast(w, dtype=x.dtype.name)
        _lns0, dxx, dyy, dzz, dxy, dyz, dxz = tf.split(x, 7, axis=-1)  # [N,Nx,Ny,1]
        phi = bvals * (gx * gx * dxx + gy * gy * dyy + gz * gz * dzz +
                       2.0 * gx * gy * dxy + 2.0 * gy * gz * dyz + 2.0 * gx * gz * dxz)  # [N,Nx,Ny,Nb]
        r = _lns0 + phi - y  # [N,Nx,Ny,Nb]

        dSds0 = tf.ones_like(r)  # [N,Nx,Ny,Nb]
        dSddxx = bvals * gx * gx
        dSddyy = bvals * gy * gy
        dSddzz = bvals * gz * gz
        dSddxy = 2.0 * bvals * gx * gy
        dSddyz = 2.0 * bvals * gy * gz
        dSddxz = 2.0 * bvals * gx * gz

        Jr = tf.stack([tf.math.reduce_mean(dSds0 * r * signal**2, axis=-1),
                       tf.math.reduce_mean(dSddxx * r * signal**2, axis=-1),
                       tf.math.reduce_mean(dSddyy * r * signal**2, axis=-1),
                       tf.math.reduce_mean(dSddzz * r * signal**2, axis=-1),
                       tf.math.reduce_mean(dSddxy * r * signal**2, axis=-1),
                       tf.math.reduce_mean(dSddyz * r * signal**2, axis=-1),
                       tf.math.reduce_mean(dSddxz * r * signal**2, axis=-1)], axis=-1)  # [N,Nx,Ny,7]

        grad = w * Jr + rho * (x + beta - z)  # gradient [N,Nx,Ny,7]

        h11 = w * tf.math.reduce_mean(dSds0 * dSds0 * signal**2, axis=-1) + rho
        h12 = w * tf.math.reduce_mean(dSds0 * dSddxx * signal**2, axis=-1)
        h13 = w * tf.math.reduce_mean(dSds0 * dSddyy * signal**2, axis=-1)
        h14 = w * tf.math.reduce_mean(dSds0 * dSddzz * signal**2, axis=-1)
        h15 = w * tf.math.reduce_mean(dSds0 * dSddxy * signal**2, axis=-1)
        h16 = w * tf.math.reduce_mean(dSds0 * dSddyz * signal**2, axis=-1)
        h17 = w * tf.math.reduce_mean(dSds0 * dSddxz * signal**2, axis=-1)
        h22 = w * tf.math.reduce_mean(dSddxx * dSddxx * signal**2, axis=-1) + rho
        h23 = w * tf.math.reduce_mean(dSddxx * dSddyy * signal**2, axis=-1)
        h24 = w * tf.math.reduce_mean(dSddxx * dSddzz * signal**2, axis=-1)
        h25 = w * tf.math.reduce_mean(dSddxx * dSddxy * signal**2, axis=-1)
        h26 = w * tf.math.reduce_mean(dSddxx * dSddyz * signal**2, axis=-1)
        h27 = w * tf.math.reduce_mean(dSddxx * dSddxz * signal**2, axis=-1)
        h33 = w * tf.math.reduce_mean(dSddyy * dSddyy * signal**2, axis=-1) + rho
        h34 = w * tf.math.reduce_mean(dSddyy * dSddzz * signal**2, axis=-1)
        h35 = w * tf.math.reduce_mean(dSddyy * dSddxy * signal**2, axis=-1)
        h36 = w * tf.math.reduce_mean(dSddyy * dSddyz * signal**2, axis=-1)
        h37 = w * tf.math.reduce_mean(dSddyy * dSddxz * signal**2, axis=-1)
        h44 = w * tf.math.reduce_mean(dSddzz * dSddzz * signal**2, axis=-1) + rho
        h45 = w * tf.math.reduce_mean(dSddzz * dSddxy * signal**2, axis=-1)
        h46 = w * tf.math.reduce_mean(dSddzz * dSddyz * signal**2, axis=-1)
        h47 = w * tf.math.reduce_mean(dSddzz * dSddxz * signal**2, axis=-1)
        h55 = w * tf.math.reduce_mean(dSddxy * dSddxy * signal**2, axis=-1) + rho
        h56 = w * tf.math.reduce_mean(dSddxy * dSddyz * signal**2, axis=-1)
        h57 = w * tf.math.reduce_mean(dSddxy * dSddxz * signal**2, axis=-1)
        h66 = w * tf.math.reduce_mean(dSddyz * dSddyz * signal**2, axis=-1) + rho
        h67 = w * tf.math.reduce_mean(dSddyz * dSddxz * signal**2, axis=-1)
        h77 = w * tf.math.reduce_mean(dSddxz * dSddxz * signal**2, axis=-1) + rho

        h1 = tf.stack([h11, h12, h13, h14, h15, h16, h17], axis=-1)
        h2 = tf.stack([h12, h22, h23, h24, h25, h26, h27], axis=-1)
        h3 = tf.stack([h13, h23, h33, h34, h35, h36, h37], axis=-1)
        h4 = tf.stack([h14, h24, h34, h44, h45, h46, h47], axis=-1)
        h5 = tf.stack([h15, h25, h35, h45, h55, h56, h57], axis=-1)
        h6 = tf.stack([h16, h26, h36, h46, h56, h66, h67], axis=-1)
        h7 = tf.stack([h17, h27, h37, h47, h57, h67, h77], axis=-1)
        hessian = tf.stack([h1, h2, h3, h4, h5, h6, h7], axis=-2)  # [N,Nx,Ny,7,7]

        hessian_inv = tf.linalg.inv(hessian)
        x = x - tf.math.reduce_sum(hessian_inv * grad[..., tf.newaxis, :], axis=-1)   # [N,Nx,Ny,7]

        # %% For non-invertible matrix
        # epsilon = 1e-08 * tf.eye(7)
        # hessian_new = tf.where(tf.linalg.det(hessian) > 0.0, 0.0, 1.0)
        # hessian_new = hessian_new[..., tf.newaxis, tf.newaxis]*epsilon + hessian
        # hessian_inv = tf.linalg.inv(hessian_new)
        # x = x - tf.math.reduce_sum(hessian_inv * grad[..., tf.newaxis, :], axis=-1)   # [N,Nx,Ny,7]
        return x


class AuxVarLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(AuxVarLayer, self).__init__(name=name)

    def call(self, x, beta, rho, lam, dz):
        rho = tf.cast(rho, dtype=x.dtype.name)
        lam = tf.cast(lam, dtype=x.dtype.name)
        z = tf.math.divide_no_nan(rho * (x + beta) + lam * dz, rho + lam)  # (ρ(x+β)+λf(z))/(ρ+λ)
        return z


class ADMMNetm(tf.keras.layers.Layer):
    def __init__(self, Ns, Nf, Na, Nblock, Nfilters, f, denoiser, recon_flag=False, bn_flag=False, name=None):
        super(ADMMNetm, self).__init__(name=name)
        self.Ns = Ns
        self.Nf = Nf
        self.Na = Na
        self.recon = recon_flag

        self.lam = tf.Variable(initial_value=0.001, trainable=True, name=name + '_lam',
                               constraint=tf.keras.constraints.NonNeg())
        self.rho = tf.Variable(initial_value=0.1, trainable=True, name=name + '_rho',
                               constraint=tf.keras.constraints.NonNeg())
        self.sigma = tf.Variable(initial_value=1.0, trainable=False, name=name + '_sigma',
                                 constraint=tf.keras.constraints.NonNeg())

        self.fittinglayer = FittingLayer(name='recon')
        self.auxvarlayer = AuxVarLayer(name='auxvar')
        if denoiser == 'DnCNN':
            self.denoiser = net_denoiser.DnCNN(Nblock=Nblock, Nfilters=Nfilters, f=f, bn_flag=bn_flag, name='DnCNN')
        if denoiser == 'ResNet':  # do not provide
            self.denoiser = net_denoiser.ResNet(Nblock=Nblock, Nfilters=Nfilters, f=f, name='ResNet')

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, inputs):
        Ni = 1
        y = inputs[0]
        bvecs = inputs[1]
        bvals = inputs[2]
        mask = inputs[3]
        weights = tf.ones_like(y)
        bvecs = bvecs[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # [N,1,1,Nb,3]
        bvals = bvals[:, tf.newaxis, tf.newaxis, tf.newaxis, :]  # [N,1,1,Nb]

        #%% INITIALIZATION method 2
        _lns0 = y[..., 0] * 0.0 + 1.0
        dxx = y[..., 0] * 0.0 + 0.001
        dyy = y[..., 0] * 0.0 + 0.001
        dzz = y[..., 0] * 0.0 + 0.001
        dxy = y[..., 0] * 0.0 + 0.001
        dyz = y[..., 0] * 0.0 + 0.001
        dxz = y[..., 0] * 0.0 + 0.001
        x = tf.stack([_lns0, dxx, dyy, dzz, dxy, dyz, dxz], axis=-1)
        x = func.range_constraint(x, mask)
        z = x
        beta = x * 0.0

        xm = []
        for i_Ni in range(Ni):
            x = self.fittinglayer(x, z, beta, 0.0, self.sigma, y, bvecs, bvals, weights, mask)
            x = func.range_constraint(x, mask)
        z = x
        beta = x * 0.0

        #%% Unrolling start
        for i_Ns in range(self.Ns):
            for i_Nf in range(self.Nf):
                weights = func.recon_lls_train(x, bvecs, bvals)
                weights = tf.exp(-1.0 * weights)
                x = self.fittinglayer(x, z, beta, self.rho, self.sigma, y, bvecs, bvals, weights, mask)
                x = func.range_constraint(x, mask)

            for i_Na in range(self.Na):
                fz = self.denoiser(z)
                z = self.auxvarlayer(x, beta, self.rho, self.lam, fz)
                z = func.range_constraint(z, mask)

            # multiplier layer
            beta = beta + x - z

            # Guiding optimization direction
            xm.append(x)    # train & test
            xm.append(fz)   # train
            xm.append(z)    # train
        xm.append(x)        # train
        # print(self.rho)
        # print(self.lam)

        if self.recon is False:
            output_map = tf.stack(xm)
            return output_map

        #%% recon back to DW image and output
        else:
            map_lls = tf.stack(xm)
            _lns0, dxx, dyy, dzz, dxy, dyz, dxz = tf.split(map_lls, 7, axis=-1)  # [stage,N,Nx,Ny,Nq]
            bvecs = bvecs[tf.newaxis, ...]  # [stage,N,1,1,Nq]
            g1 = bvecs[..., 0]
            g2 = bvecs[..., 1]
            g3 = bvecs[..., 2]
            fi = dxx * g1 * g1 + dyy * g2 * g2 + dzz * g3 * g3 + 2 * dxy * g1 * g2 + 2 * dyz * g2 * g3 + 2 * dxz * g1 * g3
            output_signal = _lns0 + bvals[tf.newaxis, ...] * fi
            # signal = tf.exp(-1.0 * signal)  # input label is lls form
            return output_signal


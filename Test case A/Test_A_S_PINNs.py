############################ S-PINN solver ##########################################
###### idea: solve the water flow equation first, then the solute equation after#### 
############################ (sequentiel training) ##################################

############################## Test case A ##########################################
#####################################################################################
#####################################################################################

# pip install deepxde

import tensorflow_probability as tfp
import tensorflow as tf
import deepxde as dde
import numpy as np
import pandas as pd
import time
import os
import random
import math


# soil geometry
soil =[0.0,-1.0,0.0,1.0] #[zmin, zmax, Tinitial, Tfinal]



# WRC: theta(psi)
def theta_function(h, thetar, thetas, alpha, n, m):
        term2 = 1 + tf.pow(-alpha * h, n)
        term3 = tf.pow(term2, -m)
        result = thetar + (thetas - thetar) * term3
        result = tf.where(h > 0, thetaSvg, result)
        return result

# HCF: K(psi)
def K_function(h, thetar, thetas, alpha, n, m, Ks):
        theta_h = theta_function(h, thetar, thetas, alpha, n, m)
        term1 = tf.pow((theta_h - thetar) / (thetas - thetar), 0.5)
        term2 = 1 - tf.pow(1 - tf.pow((theta_h - thetar) / (thetas - thetar), 1/m), m)
        result = Ks * term1 * tf.pow(term2, 2)
        result = tf.where(h > 0, ksvg, result)
        return result

#Dispersion function
def diffusion_term(theta, q, thetas, DL, Dw):
    return DL * tf.abs(q) + tf.pow(theta, 7.0/3) * Dw / tf.pow(thetas, 2.0)


# soil parameters: loam  [m and day] units
nvg= tf.constant([1.56])
mvg= 1-1/nvg
ksvg= tf.constant([0.2496])
alphavg= tf.constant([3.6])
thetaRvg= tf.constant([0.078])
thetaSvg= tf.constant(0.43, dtype=tf.float32)

# solute parameters
DL = tf.constant(0.1, dtype=tf.float32)
Dw = tf.constant(0.0, dtype=tf.float32)
rho = tf.constant(0.0, dtype=tf.float32)
Kd = tf.constant(0.0, dtype=tf.float32)



############################ IBC##############:
#water:
psi_ic = -1. # initial pressure head
q_inlet = -0.05 # upper water flux

#solute:
c_initial= 0. # # initial solute concentration
c0_exact= 50.*q_inlet # upper solute flux


#PINNs structure
num_layers = 5
num_neurons = 50
number_random = 111
layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

#iteratons for water and solute
itwater = 20000
itc = 20000


#function for generating collocation points
def  get_collocations(soil, n):
     z = np.random.uniform(soil[0], soil[1], n).reshape(-1, 1)
     t =  np.random.uniform(soil[2], soil[3], n).reshape(-1, 1)
     return t, z

# size for each collocation points
n_res,  n_ic, n_up, n_dw  =  10000, 100, 1000, 100


np.random.seed(0)

# residual points
t_res, z_res = get_collocations(soil, n_res)
# initial collocation points
t_ic,  z_ic = get_collocations(list(np.append(soil[0:3],0)), n_ic)
# collocation points for soil surface boundary
t_up, z_up = get_collocations([soil[0],soil[0],soil[2],soil[3]], n_up)
# collocation points for soil bottom boundary
t_dw, z_dw = get_collocations([soil[1],soil[1],soil[2],soil[3]], n_dw)


############################# water solver ###############################################
class water:

    def __init__(self,  layers, LAA, ):



        self.LAA = LAA

        self.weights_psi, self.biases_psi, self.A_psi = self.initialize_NN(layers)



        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf,self.t_ic_tf,  self.z_ic_tf, \
         self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))


        # prediction from PINNs
        self.psi_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.psi_ic_pred= self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.q_up_pred= self.net_q_up(self.t_up_tf, self.z_up_tf)
        self.q_dw_pred= self.net_q_dw(self.t_dw_tf, self.z_dw_tf)

        self.q_up = tf.fill(tf.shape(self.q_up_pred), q_inlet) #Upper BC
        self.psi_ic_exact = tf.fill(tf.shape(self.psi_ic_pred), psi_ic) #IC

        #weights for the loss functions
        self.constant_ic, self.constant_up, self.constant_dw, self.constant_res =  1, 100, 1, 1


        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_ic = tf.reduce_mean(tf.square(self.psi_ic_pred - self.psi_ic_exact))
        self.loss_up = tf.reduce_mean(tf.square(self.q_up_pred - self.q_up))
        self.loss_dw = tf.reduce_mean(tf.square(self.q_dw_pred))
        self.loss = self.constant_res * self.loss_res + self.constant_ic* self.loss_ic \
                 +  self.constant_up* self.loss_up \
                 +  self.constant_dw * self.loss_dw


         # L-BFGS-B method
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})



        # define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()

        # total loss
        self.loss_total = []


    def initialize_NN(self, layers):
        num_layers = len(layers)
        weights = []
        biases = []
        A = []
        for l in range(0, num_layers-1):
            in_dim = layers[l]
            out_dim = layers[l+1]
            xavier_stddev = np.sqrt(2/(in_dim + out_dim))
            W = tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev),dtype=tf.float32, trainable=True)
            b = tf.Variable(np.zeros([1, out_dim]), dtype=tf.float32, trainable=True)
            weights.append(W)
            biases.append(b)
            a = tf.Variable(0.05, dtype=tf.float32, trainable=True)
            A.append(a)
        return weights, biases, A


    def net_psi(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            # activation
            if l < num_layers-2:
                if self.LAA:
                    H = tf.tanh(20 *A[l]*H)
                else:
                    H = tf.tanh(H)
        return  -tf.exp(H)


    def net_res(self, t, z):
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)


        theta= theta_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]


        q_exact=-K*(psi_z+1)


        q_z = tf.gradients(q_exact, z)[0]

        # residual loss

        res_richards = theta_t + q_z


        return  psi, res_richards

    def net_ic(self, t, z):
        X = tf.concat([t, z],1)

        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)

        return  psi


    def net_q_up(self, t, z):
        X = tf.concat([t, z],1)


        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_z = tf.gradients(psi, z)[0]
        q=-K*(psi_z+1)

        return q

    def net_q_dw(self, t, z):
        X = tf.concat([t, z],1)


        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)
        psi_z = tf.gradients(psi, z)[0]


        return psi_z



    def net_water(self, t, z, w, b, a):
        X = tf.concat([t, z],1)

        psi = self.net_psi(X, w, b, a)
        theta =theta_function(psi, thetaRvg,thetaSvg, alphavg, nvg, mvg)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_z = tf.gradients(psi, z)[0]
        q=-K*(psi_z+1)

        return   theta, q




    def train(self, N_iter, batch = False, batch_size = 500):
        start_time = time.time()
        print("Adams epochs:")
        for it in range(N_iter):

            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, z_r) = (t_res[idx_res,:],
                                  z_res[idx_res,:])
            else:

                (t_r, z_r) = (t_res,
                                  z_res)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
                print("-" * 120)
                print('Epoch: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
                self.loss_total.append(loss_value)

        # L-BFGS-B
        print("-" * 120)
        print("L-BFGS-B epochs:")
        tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])

        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)


    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star, self.t_up_tf:  t_star, self.z_up_tf: z_star}
        psi = self.sess.run(self.psi_pred, tf_dict)
        weights_psi = self.sess.run(self.weights_psi)
        biases_psi = self.sess.run(self.biases_psi)
        a_psi = self.sess.run(self.A_psi)
        theta = self.sess.run(theta_function(psi, thetaRvg, thetaSvg, alphavg,nvg, mvg))
        q = self.sess.run(self.q_up_pred, tf_dict)
        total_loss = self.loss_total
        return psi, weights_psi, biases_psi, a_psi, theta, q, total_loss
    

tf.compat.v1.set_random_seed(0) # TensorFlow's random generator fixed
Richards = water(layers, LAA=False)

# train the water PINN solver
Richards.train(itwater)

#prediction points
nz = 1001
nt = 11
z = np.linspace(soil[0], soil[1], num=nz)
t = np.linspace(soil[2], soil[3], num=nt)
z_pred, t_pred  =np.meshgrid(z,t)

# make predictions
psi, w, b, a, theta, q, water_loss = Richards.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))

############################# solute solver ###############################################
class C:

    def __init__(self, layers, LAA):

        self.LAA = LAA

        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_ic_tf, self.z_ic_tf\
         , self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))


        # prediction from PINNs
        self.c_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.c_ic_pred = self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.qc_up_pred= self.net_qc_up(self.t_up_tf, self.z_up_tf)
        self.qc_dw_pred= self.net_qc_dw(self.t_dw_tf, self.z_dw_tf)

        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), c_initial) #IC
        self.qc_up = tf.fill(tf.shape(self.qc_up_pred), c0_exact) #Upper BC

        #weights for loss function
        self.constant_ic, self.constant_up, self.constant_dw, self.constant_res = 1, 100, 1, 1


        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_ic = tf.reduce_mean(tf.square(self.c_ic_pred - self.c_ic))
        self.loss_up = tf.reduce_mean(tf.square(self.qc_up_pred - self.qc_up))
        self.loss_dw = tf.reduce_mean(tf.square(self.qc_dw_pred))
        self.loss = self.constant_res * self.loss_res +  self.constant_ic * self.loss_ic   \
                 +  self.constant_up* self.loss_up \
                 +  self.constant_dw * self.loss_dw


         # L-BFGS-B method
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})



        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()
        self.loss_total = []


    def initialize_NN(self, layers):
        num_layers = len(layers)
        weights = []
        biases = []
        A = []
        for l in range(0, num_layers-1):
            in_dim = layers[l]
            out_dim = layers[l+1]
            xavier_stddev = np.sqrt(2/(in_dim + out_dim))
            W = tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev),dtype=tf.float32, trainable=True)
            b = tf.Variable(np.zeros([1, out_dim]), dtype=tf.float32, trainable=True)
            weights.append(W)
            biases.append(b)
            a = tf.Variable(0.05, dtype=tf.float32, trainable=True)
            A.append(a)
        return weights, biases, A



    def net_c(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            if l < num_layers-2:
                if self.LAA:
                    H = tf.tanh(20 *A[l]*H)
                else:
                    H = tf.tanh(H)
        return  H


    def net_ic(self, t, z):
        X = tf.concat([t, z],1)

        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)

        return  c

    def net_res(self, t, z):
        X = tf.concat([t, z],1)

        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c_t = tf.gradients(theta*c, t)[0]
        c_z = tf.gradients(c, z)[0]
        D =diffusion_term(theta,q, thetaSvg,DL,Dw)
        qc = tf.gradients(q*c-D*c_z, z)[0]


        fc=   c_t + qc

        return  c, fc

    def net_qc_up(self, t, z):
        X = tf.concat([t, z],1)


        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c_z = tf.gradients(c, z)[0]
        D =diffusion_term(theta,q, thetaSvg, DL, Dw)
        qc = q*c-D*c_z

        return  qc

    def net_qc_dw(self, t, z):
        X = tf.concat([t, z],1)


        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        c_z = tf.gradients(c, z)[0]

        return c_z

    def net_c1(self, t, z, w, b, a):
        X = tf.concat([t, z],1)

        c = self.net_c(X, w, b, a)

        return  c


    def train(self, N_iter, batch = False, batch_size = 500):
        start_time = time.time()
        print("Adams epochs:")
        for it in range(N_iter):

            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, z_r) = (t_res[idx_res,:],
                                  z_res[idx_res,:])
            else:

                (t_r, z_r) = (t_res,
                                  z_res)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
                print("-" * 120)
                print('Epoch: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
                self.loss_total.append(loss_value)

        # L-BFGS-B
        print("-" * 120)
        print("L-BFGS-B epochs:")
        tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])

        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)


    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)
        total_loss = self.loss_total
        return c, weights_c, biases_c, a_c, total_loss
    

solute = C(layers, LAA=False)
# train the solute PINN solver
solute.train(itc)

# make predictions
c, wc, bc, ac, total_loss_c = solute.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))
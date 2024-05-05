############################ S-PINN solver ##########################################
############################## Test case C: 2D water-solute ##########################
#####################################################################################
#####################################################################################



## BEGIN OF THE PROGRAM ####

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
soil =[-1., 1.,0.0, -1., 0.0, 1.] #[xmin, xmax, zmin,zmax, Tinitial, Tfinal]

# VG parameters : loam  m and day
nvg= tf.constant([1.56])
mvg= 1-1/nvg
ksvg= tf.constant([0.25])
alphavg= tf.constant([3.6]) 
thetaRvg= tf.constant([0.078])
thetaSvg= tf.constant(0.43, dtype=tf.float32)
        
# solute parameters 
DL = tf.constant(0.5, dtype=tf.float32)
DT = tf.constant(0.1, dtype=tf.float32)
Dw = tf.constant(0.0, dtype=tf.float32)        
        

#IC and BC
psi_initial = -1.3
psi_inlet = -0.2
c_initial = 0.1
c_inlet = 1.0


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

#PINNs structure
num_layers = 5
num_neurons = 50
number_random = 111
layers = np.concatenate([[3], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

#iteratons for water and solute
itwater = 20000
itc = 20000

def  get_collocations(soil, n):
    x= np.random.uniform(soil[0], soil[1], n).reshape(-1, 1)
    z = np.random.uniform(soil[2], soil[3], n).reshape(-1, 1)
    t =  np.random.uniform(soil[4], soil[5], n).reshape(-1, 1)
    return t, x, z     

n_res, n_ic, n_up, n_lb, n_left,  n_right = 10000, 100, 1000, 100, 100, 100
t_res, x_res, z_res = get_collocations(soil, n_res)
t_ic, x_ic, z_ic = get_collocations(list(np.append(soil[0:5],0)), n_ic)
t_ub, x_ub, z_ub = get_collocations([soil[0],soil[1],soil[2],soil[2],soil[4],soil[5]], n_up)
t_lb, x_lb, z_lb = get_collocations([soil[0],soil[1],soil[3],soil[3],soil[4],soil[5]], n_lb)
t_left, x_left, z_left = get_collocations([soil[0],soil[0],soil[2],soil[3],soil[4],soil[5]], n_left)
t_right, x_right, z_right = get_collocations([soil[1],soil[1],soil[2],soil[3],soil[4],soil[5]], n_right)   


def dispersion(theta, qx, qz, thetas, DL, DT, Dw):
    normq = tf.math.sqrt(qx**2 + qz**2)
    Dx = (DL * qx**2 + DT * qz**2)/normq + tf.pow(theta, 10.0/3) * Dw / tf.pow(thetas, 2.0)
    Dz = (DL * qz**2 + DT * qx**2)/normq + tf.pow(theta, 10.0/3) * Dw / tf.pow(thetas, 2.0)
    Dxz = (DL-DT)*qz*qx/normq 
    return Dx, Dxz, Dz

# prediction time-space points
meshx=101
meshz=101
    
x = np.linspace(soil[0], soil[1], num=meshx).reshape(-1, 1)
z = np.linspace(soil[2], soil[3], num=meshz).reshape(-1, 1)
t = np.linspace(soil[4], soil[5], num=5).reshape(-1, 1)
    
#algorithm of organization:    
LL=x.size*z.size*t.size
x_pred=np.empty(LL).reshape(-1, 1)
z_pred=np.empty(LL).reshape(-1, 1)
t_pred=np.empty(LL).reshape(-1, 1)
l=0
for i in range(t.size):
    t_pred[i*(x.size*z.size):(i+1)*(x.size*z.size)]=t[i]
    for j in range(x.size):
        x_pred[l*z.size:(l+1)*z.size]=x[j]
        z_pred[l*z.size:(l+1)*z.size]=z
        l=l+1 
########## water solver ##############
class water:
   
    def __init__(self, layers, LAA):

        self.LAA = LAA
        
       
        self.layers = layers   
        self.weights_psi, self.biases_psi, self.A_psi = self.initialize_NN(self.layers)


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        
       

        
        
        # tf placeholder : empty variables
        [self.t_res_tf, self.x_res_tf, self.z_res_tf,self.t_ic_tf, self.x_ic_tf,self.z_ic_tf\
         , self.t_ub_tf, self.x_ub_tf, self.z_ub_tf, self.t_lb_tf, self.x_lb_tf, self.z_lb_tf,\
         self.t_leftb_tf, self.x_leftb_tf, self.z_leftb_tf, self.t_rightb_tf, self.x_rightb_tf, self.z_rightb_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(18)] 
          
        
        
        # prediction from PINNs
        self.psi_pred,  self.psi_res_pred= self.net_res(self.t_res_tf, self.x_res_tf, self.z_res_tf)
      
        self.psi_ic_pred = self.net_ic(self.t_ic_tf, self.x_ic_tf, self.z_ic_tf)
        
        self.q_lb_pred = self.net_dw(self.t_lb_tf, self.x_lb_tf,self.z_lb_tf)
        
        self.q_leftb_pred = self.net_flux_lr(self.t_leftb_tf, self.x_leftb_tf,self.z_leftb_tf)
        self.q_rightb_pred = self.net_flux_lr(self.t_rightb_tf, self.x_rightb_tf,self.z_rightb_tf)
        
        self.psi_up_pred = self.net_ic(self.t_ub_tf, self.x_ub_tf, self.z_ub_tf)

        self.psi_ic = tf.fill(tf.shape(self.psi_ic_pred), psi_initial) #IC
        self.psi_ub = tf.fill(tf.shape(self.psi_up_pred), psi_inlet) #up BC
        
       
        # loss functions
        
        #Residual
        self.loss_res =  tf.reduce_mean(tf.square(self.psi_res_pred))
        #IC
        self.loss_ic = tf.reduce_mean(tf.square(self.psi_ic_pred - self.psi_ic))

        #lower bc
        self.loss_lb = tf.reduce_mean(tf.square(self.q_lb_pred))
        #left bc
        self.loss_leftb = tf.reduce_mean(tf.square(self.q_leftb_pred)) 
        #right bc
        self.loss_rightb = tf.reduce_mean(tf.square(self.q_rightb_pred))

        #upper bc
        self.loss_up = tf.reduce_mean(tf.square(self.psi_up_pred- self.psi_ub))

        
       
        self.loss = self.loss_res +   self.loss_ic +  \
                 self.loss_lb +  self.loss_leftb+self.loss_rightb \
                 + self.loss_up
        
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
        

    def xavier_init(self, s):
        in_dim = s[0]
        out_dim = s[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)

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
            a = tf.Variable(0.05, dtype=tf.float32)
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
                    H = tf.tanh(20 * A[l]*H)
                else:
                    H = tf.tanh(H)
        return -tf.exp(H)
    
    def net_ic(self, t, x, z):  
        X = tf.concat([t, x, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)
        
        return  psi

    def net_res(self, t, x, z):
        X = tf.concat([t, x, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)


        theta= theta_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)

        theta_t = tf.gradients(theta, t)[0]
        psi_x = tf.gradients(psi, x)[0]
        psi_z = tf.gradients(psi, z)[0]

        qx=-K*psi_x
        qz=-K*(psi_z+1)


        q_z = tf.gradients(qz, z)[0]
        q_x = tf.gradients(qx, x)[0]


        # residual loss

        res_richards = theta_t + q_z + q_x


        return  psi, res_richards
    
    
    def net_flux_lr(self, t, x, z): 
        X = tf.concat([t, x, z],1)
    
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_x = tf.gradients(psi, x)[0]

        qx=-K*psi_x
        return  qx
    
    
    
    def net_dw(self, t, x, z): 
        X = tf.concat([t, x, z],1)
   
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)

        psi_z = tf.gradients(psi, z)[0]
   

        return  psi_z

    def net_water(self, t, x, z, w, b, a):
        X = tf.concat([t, x, z],1)

        psi = self.net_psi(X, w, b, a)
        theta =theta_function(psi, thetaRvg,thetaSvg, alphavg, nvg, mvg)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_z = tf.gradients(psi, z)[0]
        psi_x = tf.gradients(psi, x)[0]

        qx=-K*psi_x
        qz=-K*(psi_z+1)

        return   theta, qx, qz
    
    
    def train(self, N_iter, batch = True, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, x_r, z_r) = (t_res[idx_res,:], x_res[idx_res,:],
                                  z_res[idx_res,:])
            else:
               
                (t_r, x_r, z_r) = (t_res, x_res, z_res)
                
            tf_dict = {self.t_res_tf: t_r,
                       self.x_res_tf: x_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.x_ic_tf: x_ic,
                       self.z_ic_tf: z_ic,
                       self.t_lb_tf: t_lb,
                       self.x_lb_tf: x_lb,
                       self.z_lb_tf: z_lb,
                       self.t_leftb_tf: t_left,
                       self.x_leftb_tf: x_left,
                       self.z_leftb_tf: z_left,
                       self.t_rightb_tf: t_right,
                       self.x_rightb_tf: x_right,
                       self.z_rightb_tf:  z_right,
                       self.t_ub_tf: t_ub, self.x_ub_tf: x_ub, self.z_ub_tf: z_ub}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' % (it, loss_value))
         

        # L-BFGS-B
        tf_dict = {self.t_res_tf: t_res,
                       self.x_res_tf: x_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.x_ic_tf: x_ic,
                       self.z_ic_tf: z_ic,
                       self.t_lb_tf: t_lb,
                       self.x_lb_tf: x_lb,
                       self.z_lb_tf: z_lb,
                       self.t_leftb_tf: t_left,
                       self.x_leftb_tf: x_left,
                       self.z_leftb_tf: z_left,
                       self.t_rightb_tf: t_right,
                       self.x_rightb_tf: x_right,
                       self.z_rightb_tf:  z_right,
                       self.t_ub_tf: t_ub, self.x_ub_tf: x_ub, self.z_ub_tf: z_ub}

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)

    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, x_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.x_res_tf: x_star,
                   self.z_res_tf: z_star}
        psi = self.sess.run(self.psi_pred, tf_dict)
        weights_psi = self.sess.run(self.weights_psi)
        biases_psi = self.sess.run(self.biases_psi)
        a_psi = self.sess.run(self.A_psi)

        theta = self.sess.run(theta_function(psi, thetaRvg, thetaSvg, alphavg,nvg, mvg))
        return psi, theta, weights_psi, biases_psi, a_psi

Richards = water(layers, LAA=True)  

Richards.train(itwater)


# train the Richards solver
Richards.train(itwater)

# predict, the pressure head, water content, and the PINN parameters
psi, theta, w, b, a = Richards.predict(t_pred, x_pred, z_pred)

########## solute solver ##############

class C:

    def __init__(self, layers, LAA):
      
        self.LAA = LAA


        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf, self.x_res_tf, self.z_res_tf,self.t_ic_tf, self.x_ic_tf,self.z_ic_tf\
         , self.t_ub_tf, self.x_ub_tf, self.z_ub_tf, self.t_lb_tf, self.x_lb_tf, self.z_lb_tf,\
         self.t_leftb_tf, self.x_leftb_tf, self.z_leftb_tf, self.t_rightb_tf, self.x_rightb_tf, self.z_rightb_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(18)] 
          

        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))

    
        # prediction from PINNs
        self.c_pred, self.res_pred = self.net_res(self.t_res_tf, self.x_res_tf, self.z_res_tf)
        
        self.c_ic_pred = self.net_ic(self.t_ic_tf, self.x_ic_tf, self.z_ic_tf)
        
        self.q_lb_pred = self.net_dw(self.t_lb_tf, self.x_lb_tf,self.z_lb_tf)
        
        self.q_leftb_pred = self.net_flux_lr(self.t_leftb_tf, self.x_leftb_tf,self.z_leftb_tf)
        self.q_rightb_pred = self.net_flux_lr(self.t_rightb_tf, self.x_rightb_tf,self.z_rightb_tf)
        
        self.c_up_pred = self.net_ic(self.t_ub_tf, self.x_ub_tf, self.z_ub_tf)
        
        
        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), c_initial) #IC
        self.c_ub = tf.fill(tf.shape(self.c_up_pred), c_inlet) #up BC
       
        # loss functions
        
        #Residual
        self.loss_res =  tf.reduce_mean(tf.square(self.res_pred))
        #IC
        self.loss_ic = tf.reduce_mean(tf.square(self.c_ic_pred - self.c_ic))

        #lower bc
        self.loss_lb = tf.reduce_mean(tf.square(self.q_lb_pred))

        #left bc
        self.loss_leftb = tf.reduce_mean(tf.square(self.q_leftb_pred)) 
 
        #right bc
        self.loss_rightb = tf.reduce_mean(tf.square(self.q_rightb_pred))
    
        #upper bc
        self.loss_up = tf.reduce_mean(tf.square(self.c_up_pred- self.c_ub))

        
       
        self.loss =self.loss_res + self.loss_ic   \
                 +  self.loss_lb +  self.loss_leftb+self.loss_rightb \
                 + self.loss_up
        
       

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
            a = tf.Variable(0.05, dtype=tf.float32)
            A.append(a)
        return weights, biases, A


    def net_c(self, X, weights, biases, A):
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
        return  H


    def net_res(self, t, x, z):
        X = tf.concat([t, x, z],1)

        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        
        theta, qx, qz = Richards.net_water(t, x, z, w, b, a)
        c_t = tf.gradients(theta*c, t)[0]
        c_x = tf.gradients(c, x)[0]
        c_z = tf.gradients(c, z)[0]         

        Dx, Dxz, Dz = dispersion(theta, qx, qz, thetaSvg, DL, DT, Dw)

        qc_x = c*qx - Dx*c_x - Dxz*c_z
        qc_z = c*qz - Dz*c_z - Dxz*c_x

        qc_x_x = tf.gradients(qc_x, x)[0]
        qc_z_z = tf.gradients(qc_z, z)[0]

        res_c = c_t + qc_x_x + qc_z_z 


        return  c, res_c
    
    def net_ic(self, t, x, z):
        X = tf.concat([t, x, z],1)

        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        return  c
    
    
    def net_flux_lr(self, t, x, z): 
        X = tf.concat([t, x, z],1)
            
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        
        theta, qx, qz = Richards.net_water(t, x, z, w, b, a)
        c_x = tf.gradients(c, x)[0]
        c_z = tf.gradients(c, z)[0]         

        Dx, Dxz, Dz = dispersion(theta, qx, qz, thetaSvg, DL, DT, Dw)

        qc_x =  Dx*c_x

        return qc_x
       
    
    def net_dw(self, t, x, z): 
        X = tf.concat([t, x, z],1)
            
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        c_z = tf.gradients(c, z)[0]         

        return c_z





    def train(self, N_iter, batch = True, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, x_r, z_r) = (t_res[idx_res,:], x_res[idx_res,:],
                                  z_res[idx_res,:])
            else:
               
                (t_r, x_r, z_r) = (t_res, x_res, z_res)
                
            tf_dict = {self.t_res_tf: t_r,
                       self.x_res_tf: x_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.x_ic_tf: x_ic,
                       self.z_ic_tf: z_ic,
                       self.t_lb_tf: t_lb,
                       self.x_lb_tf: x_lb,
                       self.z_lb_tf: z_lb,
                       self.t_leftb_tf: t_left,
                       self.x_leftb_tf: x_left,
                       self.z_leftb_tf: z_left,
                       self.t_rightb_tf: t_right,
                       self.x_rightb_tf: x_right,
                       self.z_rightb_tf:  z_right,
                       self.t_ub_tf: t_ub, self.x_ub_tf: x_ub, self.z_ub_tf: z_ub}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' % (it, loss_value))
         

        # L-BFGS-B
        tf_dict = {self.t_res_tf: t_res,
                       self.x_res_tf: x_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.x_ic_tf: x_ic,
                       self.z_ic_tf: z_ic,
                       self.t_lb_tf: t_lb,
                       self.x_lb_tf: x_lb,
                       self.z_lb_tf: z_lb,
                       self.t_leftb_tf: t_left,
                       self.x_leftb_tf: x_left,
                       self.z_leftb_tf: z_left,
                       self.t_rightb_tf: t_right,
                       self.x_rightb_tf: x_right,
                       self.z_rightb_tf:  z_right,
                       self.t_ub_tf: t_ub, self.x_ub_tf: x_ub, self.z_ub_tf: z_ub}

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)
   

    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, x_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.x_res_tf: x_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        return c


solute = C(layers, LAA=True)   

solute.train(itc)
c = solute.predict(t_pred, x_pred, z_pred)
# ! pip install deepxde

import tensorflow_probability as tfp
import matplotlib.tri as tri
import deepxde as dde
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import random
import math


class PINN:
   
    def __init__(self, soil, size, layers, LAA):
       #nb of collocation points: size = [res, ic, leftbc, rightbc, upbc, dwbc]
        self.n_res= size[0]
        self.n_ic=size[1]
        self.n_left=size[2]
        self.n_right=size[3]
        self.n_ub_center= size[4]
        self.n_ub_right= size[4]
        self.n_ub_left= size[4]
        self.n_lb= size[5]
        
        self.LAA = LAA
        
       
        self.layers = layers    
        self.weights, self.biases, self.A = self.initialize_NN(layers)

        # data
        self.t_res, self.x_res, self.z_res = self.get_collocations(soil, self.n_res)
        self.t_ic, self.x_ic, self.z_ic = self.get_collocations(list(np.append(soil[0:5],0)), self.n_ic)
        self.t_ub, self.x_ub, self.z_ub = self.get_collocations([soil[0],soil[1],soil[2],soil[2],soil[4],soil[5]], self.n_ub_left)
        self.t_lb, self.x_lb, self.z_lb = self.get_collocations([soil[0],soil[1],soil[3],soil[3],soil[4],soil[5]], self.n_lb)
        self.t_left, self.x_left, self.z_left = self.get_collocations([soil[0],soil[0],soil[2],soil[3],soil[4],soil[5]], self.n_left)
        self.t_right, self.x_right, self.z_right = self.get_collocations([soil[1],soil[1],soil[2],soil[3],soil[4],soil[5]], self.n_right)
        

        
        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        
        # VG parameters : loam  m and day
        self.nvg= tf.constant([1.56])
        self.mvg= 1-1/self.nvg
        self.ksvg= tf.constant([0.25])
        self.alphavg= tf.constant([3.6]) 
        self.thetaRvg= tf.constant([0.078])
        self.thetaSvg= tf.constant(0.43, dtype=tf.float32)
        
        # solute parameters 
        self.DL = tf.constant(0.5, dtype=tf.float32)
        self.DT = tf.constant(0.1, dtype=tf.float32)
        self.Dw = tf.constant(0.0, dtype=tf.float32)
        

        
        #IC and BC
        self.psi_initial = -1.3
        self.psi0_inlet = -0.2
        self.c_initial = 0.1
        self.c0_inlet = 1.0

        
        
        # tf placeholder : empty variables
        [self.t_res_tf, self.x_res_tf, self.z_res_tf,self.t_ic_tf, self.x_ic_tf,self.z_ic_tf\
         , self.t_ub_tf, self.x_ub_tf, self.z_ub_tf, self.t_lb_tf, self.x_lb_tf, self.z_lb_tf,\
         self.t_leftb_tf, self.x_leftb_tf, self.z_leftb_tf, self.t_rightb_tf, self.x_rightb_tf, self.z_rightb_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(18)] 
          
        
        
        # prediction from PINNs
        self.psi_pred, self.c_pred, self.psi_res_pred, self.c_res_pred = self.net_res(self.t_res_tf, self.x_res_tf, self.z_res_tf)
        
        self.psi_ic_pred, self.c_ic_pred = self.net_ic(self.t_ic_tf, self.x_ic_tf, self.z_ic_tf)
        
        self.q_lb_pred, self.qc_lb_pred = self.net_dw(self.t_lb_tf, self.x_lb_tf,self.z_lb_tf)
        
        self.q_leftb_pred, self.qc_leftb_pred = self.net_flux_lr(self.t_leftb_tf, self.x_leftb_tf,self.z_leftb_tf)
        self.q_rightb_pred, self.qc_rightb_pred = self.net_flux_lr(self.t_rightb_tf, self.x_rightb_tf,self.z_rightb_tf)
        
        self.psi_up_pred, self.c_up_pred = self.net_ic(self.t_ub_tf, self.x_ub_tf, self.z_ub_tf)
        
        #weights for loss function
        self.constant_ic, self.constant_ub, self.constant_lb, self.constant_res, self.constant_lr = 1, 10, 1, 1, 1
        #  5, 10, 1, 1, 1 1
        
        self.psi_ic = tf.fill(tf.shape(self.psi_ic_pred), self.psi_initial) #IC
        self.psi_ub = tf.fill(tf.shape(self.psi_up_pred), self.psi0_inlet) #up BC
        
        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), self.c_initial) #IC
        self.c_ub = tf.fill(tf.shape(self.c_up_pred), self.c0_inlet) #up BC
       
        # loss functions
        
        #Residual
        self.loss_res =  tf.reduce_mean(tf.square(self.psi_res_pred) + tf.square(self.c_res_pred))
        #IC
        self.loss_ic_R = tf.reduce_mean(tf.square(self.psi_ic_pred - self.psi_ic))
        self.loss_ic_C = tf.reduce_mean(tf.square(self.c_ic_pred - self.c_ic))
        self.loss_ic = self.loss_ic_R + self.loss_ic_C
        #lower bc
        self.loss_lb_R = tf.reduce_mean(tf.square(self.q_lb_pred))
        self.loss_lb_C =  tf.reduce_mean(tf.square(self.qc_lb_pred))
        self.loss_lb = self.loss_lb_R + self.loss_lb_C
        #left bc
        self.loss_leftb_R = tf.reduce_mean(tf.square(self.q_leftb_pred)) 
        self.loss_leftb_C =  tf.reduce_mean(tf.square(self.qc_leftb_pred))
        self.loss_leftb = self.loss_leftb_R + self.loss_leftb_C
        #right bc
        self.loss_rightb_R = tf.reduce_mean(tf.square(self.q_rightb_pred))
        self.loss_rightb_C =  tf.reduce_mean(tf.square(self.qc_rightb_pred))
        self.loss_rightb = self.loss_rightb_R + self.loss_rightb_C
        #upper bc
        self.loss_up_R = tf.reduce_mean(tf.square(self.psi_up_pred- self.psi_ub))
        self.loss_up_C = tf.reduce_mean(tf.square(self.c_up_pred- self.c_ub))
        self.loss_up = self.loss_up_R + self.loss_up_C 
        
       
        self.loss = self.constant_res * self.loss_res +  self.constant_ic * self.loss_ic   \
                 +  self.constant_lb * self.loss_lb +  self.constant_lr * (self.loss_leftb+self.loss_rightb) \
                 + self.constant_ub *self.loss_up
        
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
        
    def  get_collocations(self, soil, n):
        x= np.random.uniform(soil[0], soil[1], n).reshape(-1, 1)
        z = np.random.uniform(soil[2], soil[3], n).reshape(-1, 1)
        t =  np.random.uniform(soil[4], soil[5], n).reshape(-1, 1)
        return t, x, z     

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

    def net_coupled(self, X, weights, biases, A): 
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
        psi, c = tf.split(H  , 2, axis=1)
        return -tf.exp(psi), c
    
    def net_ic(self, t, x, z):  
        X = tf.concat([t, x, z],1)

        psi, c = self.net_coupled(X, self.weights, self.biases, self.A)
        return  psi, c

    def net_res(self, t, x, z): 
        X = tf.concat([t, x, z],1)
        if self.normalize:
            X0 = np.concatenate(t,z, 1)
            self.X_mean = X0.mean(0, keepdims=True)
            self.X_std = X0.std(0, keepdims=True)
            X = (X - self.X_mean)/self.X_std
            
        psi, c = self.net_coupled(X, self.weights, self.biases, self.A)

        theta= self.theta_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg)
        K= self.K_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg, self.ksvg)
        
        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_x = tf.gradients(psi, x)[0]
        qx=-K*psi_x
        qz=-K*(psi_z+1)
        psi_zz = tf.gradients(psi_z, z)[0]
        psi_xx = tf.gradients(psi_x, x)[0]
        K_z = tf.gradients(K, z)[0]
        K_x = tf.gradients(K, x)[0]
        
        # Richards residual loss
        f = theta_t - K_z*psi_z- K*psi_zz - K_x*psi_x- K*psi_xx - K_z
        
        c_t = tf.gradients(c, t)[0]
        c_x = tf.gradients(c, x)[0]
        c_xx = tf.gradients(c_x, x)[0]
        c_z = tf.gradients(c, z)[0]
        c_zz = tf.gradients(c_z, z)[0]
        c_xz = tf.gradients(c_x, z)[0]
        c_zx = tf.gradients(c_z, x)[0]
        
        Dx, Dxz, Dz = self.dispersion(theta, qx, qz, self.thetaSvg, self.DL, self.DT, self.Dw)       
        Dx_x =  tf.gradients(Dx, x)[0]
        Dz_z =  tf.gradients(Dz, z)[0]
        Dxz_x = tf.gradients(Dxz, x)[0]
        Dxz_z = tf.gradients(Dxz, z)[0]
        
        # Solute residual loss
        fc = theta*c_t + qx*c_x + qz*c_z - Dx*c_xx - Dxz*(c_xz+c_zx) - Dz*c_zz - c_x*(Dx_x+Dxz_z) - c_z*(Dz_z + Dxz_x)

        return  psi, c, f, fc
    
    
    def net_flux_lr(self, t, x, z): 
        X = tf.concat([t, x, z],1)

        psi, c = self.net_coupled(X, self.weights, self.biases, self.A)

        theta= self.theta_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg)
        K= self.K_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg, self.ksvg)
        psi_z = tf.gradients(psi, z)[0]
        psi_x = tf.gradients(psi, x)[0]
        qx=-K*psi_x
        qz=-K*(psi_z+1)
        
        Dx, Dxz, Dz = self.dispersion(theta, qx, qz, self.thetaSvg, self.DL, self.DT, self.Dw)
        
        c_x = tf.gradients(c, x)[0]
        c_z = tf.gradients(c, z)[0]
        
        qc=  - Dxz*c_z - Dx*c_x + qx*c      
        return  qx, qc
    
    
    
    def net_dw(self, t, x, z): 
        X = tf.concat([t, x, z],1)
   
        psi, c = self.net_coupled(X, self.weights, self.biases, self.A)

        psi_z = tf.gradients(psi, z)[0]
        c_z = tf.gradients(c, z)[0]     

        return  psi_z, c_z

      

    def theta_function(self, h, thetar, thetas, alpha, n, m): 
        term2 = 1 + tf.pow(-alpha * h, n)
        term3 = tf.pow(term2, -m)
        result = thetar + (thetas - thetar) * term3
        result = tf.where(h > 0, self.thetaSvg, result)
        return result

    def K_function(self, h, thetar, thetas, alpha, n, m, Ks):
        theta_h = self.theta_function(h, thetar, thetas, alpha, n, m)
        term1 = tf.pow((theta_h - thetar) / (thetas - thetar), 0.5)
        term2 = 1 - tf.pow(1 - tf.pow((theta_h - thetar) / (thetas - thetar), 1/m), m)
        result = Ks * term1 * tf.pow(term2, 2)
        result = tf.where(h > 0, self.ksvg, result)
        return result

    def dispersion(self, theta, qx, qz, thetas, DL, DT, Dw):
        normq = tf.math.sqrt(qx**2 + qz**2)
        Dx = (DL * qx**2 + DT * qz**2)/normq + tf.pow(theta, 10.0/3) * Dw / tf.pow(thetas, 2.0)
        Dz = (DL * qz**2 + DT * qx**2)/normq + tf.pow(theta, 10.0/3) * Dw / tf.pow(thetas, 2.0)
        Dxz = (DL-DT)*qz*qx/normq 
     
        return Dx, Dxz, Dz
    
    
    def train(self, N_iter):
        tf_dict = {self.t_res_tf: self.t_res,
                       self.x_res_tf: self.x_res,
                       self.z_res_tf: self.z_res,
                       self.t_ic_tf: self.t_ic,
                       self.x_ic_tf: self.x_ic,
                       self.z_ic_tf: self.z_ic,
                       self.t_lb_tf: self.t_lb,
                       self.x_lb_tf: self.x_lb,
                       self.z_lb_tf: self.z_lb,
                       self.t_leftb_tf: self.t_left,
                       self.x_leftb_tf:self.x_left,
                       self.z_leftb_tf:self.z_left,
                       self.t_rightb_tf:self.t_right, 
                       self.x_rightb_tf:self.x_right, 
                       self.z_rightb_tf: self.z_right,
                       self.t_ub_tf:self.t_ub, self.x_ub_tf:self.x_ub, self.z_ub_tf:self.z_ub}
       
        
        
        start_time = time.time()
        # Adam
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict) 
         # prints the iteration number, loss value, and elapsed time every 10 iterations
       
            
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_ub_R_value, loss_ubC_value, \
                loss_lb_value, loss_leftb_value, loss_rightb_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up_R, self.loss_up_C, self.loss_lb, self.loss_leftb, self.loss_rightb], tf_dict)
                
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_upR: %.3e, Loss_upC: %.3e, Loss_lb: %.3e, Loss_left: %.3e, Loss_right: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_ub_R_value, loss_ubC_value,  loss_lb_value, loss_leftb_value, loss_rightb_value, elapsed))
                start_time = time.time()

         
        
        # L-BFGS-B
    
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
        c = self.sess.run(self.c_pred, tf_dict)
    
        return psi, c
    


def main(soil, size , num_layers, num_neurons, number_random, it):

    # reset the graph and set random seeds
    tf.compat.v1.reset_default_graph() # clear all (equivalent in MATLAB)
    tf.compat.v1.set_random_seed(0) # TensorFlow's random generator fixed
    random.seed(0) # Python's random generator fixed
    np.random.seed(0) # NumPy's random generator fixed

    

    layers = np.concatenate([[3], num_neurons*np.ones(num_layers), [2]]).astype(int).tolist()


    # random seeds
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(number_random)
    random.seed(number_random)
    np.random.seed(number_random)


    model = PINN(soil, size, layers, LAA=True, monotonic=False, normalize = False) 
    #train
    model.train(it)


    #prediction
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
    
    
    psi_pred, c_pred  = model.predict(t_pred, x_pred, z_pred)
    
    pred_data = pd.DataFrame({'z': z_pred.flatten(), 't': t_pred.flatten(), 'x': x_pred.flatten(),
                         'psi_pred': psi_pred.flatten(),
                         'c_pred': c_pred.flatten()})
    pred_data.to_csv(f"./Results/Test_C_PINNs_adaptive_activation_predicted_data.csv")



soil =[-1., 1.,0.0, -1., 0.0, 1.] #[xmin, xmax, zmin,zmax, Tinitial, Tfinal]
size = [10000, 100, 100, 100, 1000, 100]  #nb of collocation points: [res, ic, leftbc, rightbc, upbc, dwbc][5000, 100, 40, 40, 1000, 40]
num_layers = [5,6,8]
num_neurons = [50,10,15]
number_random = [111]

main(soil, size, num_layers[0], num_neurons[0], number_random[0], 20000)
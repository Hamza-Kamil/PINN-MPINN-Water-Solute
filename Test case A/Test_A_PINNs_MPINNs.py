import tensorflow_probability as tfp
import deepxde as dde
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import random
import math


# PINN-MPINNs class
class PINN_MPINN:

    def __init__(self, soil, size, layers_pinns, layers_mpinns, LAA):

        
        #nb of collocation points: size = [residual:res, initial condition:ic, upbc:upper boundry, dwbc: lower boundary]
        self.n_res= size[0]
        self.nz_ub=size[2]
        self.nz_lb=size[3]
        self.nz_ic=size[1]
        self.LAA = LAA

        # data
        self.t_res, self.z_res = self.get_collocations(soil, self.n_res)
        self.t_ic, self.z_ic = self.get_collocations(list(np.append(soil[0:3],0)), self.nz_ic)
        self.t_ub, self.z_ub = self.get_collocations([soil[0],soil[0],soil[2],soil[3]], self.nz_ub)
        self.t_lb, self.z_lb = self.get_collocations([soil[1],soil[1],soil[2],soil[3]], self.nz_lb)

        # the structure of the two neural networks

        self.layers_pinns = layers_pinns
        self.layers_mpinns = layers_mpinns    
        
        self.weights_pinns, self.biases_pinns, self.A_pinns = self.initialize_NN(layers_pinns)
        self.weights_psi, self.biases_psi, self.A_psi = self.initialize_NN(layers_mpinns)
        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers_mpinns) 
        
        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))

        
        # VG parameters : loam  m and day
        self.nvg= tf.constant([1.56])
        self.mvg= 1-1/self.nvg
        self.ksvg= tf.constant([0.2496])
        self.alphavg= tf.constant([3.6]) 
        self.thetaRvg= tf.constant([0.078])
        self.thetaSvg= tf.constant(0.43, dtype=tf.float32)
        
        # solute parameters  
        self.DL = tf.constant(0.1, dtype=tf.float32)
        self.Dw = tf.constant(0.0, dtype=tf.float32)
        self.rho = tf.constant(0.0, dtype=tf.float32)
        self.Kd = tf.constant(0.0, dtype=tf.float32)
        
         # tf placeholder : empty variables
        [self.t_res_tf, self.z_res_tf,self.t_ic_tf, self.z_ic_tf\
         ,self.t_ub_tf, self.z_ub_tf,self.t_lb_tf,  self.z_lb_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)] 
          

        
        
        # prediction from PINNs
        self.psi_pred, self.psi_res_pred, self.c_pred, self.c_res_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.psi_ic_pred, self.c_ic_pred = self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.q_up_pred, self.qc_up_pred  = self.net_up(self.t_ub_tf, self.z_ub_tf)
        self.q_dw_pred, self.qc_dw_pred = self.net_dw(self.t_lb_tf, self.z_lb_tf)
        
        #weights for loss function
        self.constant_ic, self.constant_ub, self.constant_lb, self.constant_res =  10, 10, 10, 1 

        self.psi_initial= -1.
        self.c_initial= 0.
        self.q_exact= - 0.05
        self.c0_exact= 50.*self.q_exact
        
        self.psi_ic = tf.fill(tf.shape(self.psi_ic_pred), self.psi_initial) #IC
        self.q_ub = tf.fill(tf.shape(self.q_up_pred), self.q_exact) #up BC
        
        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), self.c_initial) #IC
        self.qc_ub = tf.fill(tf.shape(self.qc_up_pred), self.c0_exact) #up BC
        
        
        # loss function 
        self.loss_res =  tf.reduce_mean(tf.square(self.psi_res_pred) + tf.square(self.c_res_pred))
        self.loss_ic_R = tf.reduce_mean(tf.square(self.psi_ic_pred - self.psi_ic))
        self.loss_ic_C = tf.reduce_mean(tf.square(self.c_ic_pred - self.c_ic))
        self.loss_ic = self.loss_ic_R + self.loss_ic_C
        self.loss_lb_R = tf.reduce_mean(tf.square(self.q_dw_pred))
        self.loss_lb_C =  tf.reduce_mean(tf.square(self.qc_dw_pred))
        self.loss_lb = self.loss_lb_R + self.loss_lb_C
        self.loss_ub_R = tf.reduce_mean(tf.square(self.q_up_pred - self.q_ub))/(self.q_exact**2) 
        self.loss_ub_C = tf.reduce_mean(tf.square(self.qc_up_pred- self.qc_ub))/(self.c0_exact**2) 
        self.loss_ub = self.loss_ub_R + self.loss_ub_C
        self.loss = self.constant_res * self.loss_res +  self.constant_ic * self.loss_ic   \
                 +  self.constant_ub* self.loss_ub \
                 +  self.constant_lb * self.loss_lb 
                                                                          
        
        # L-BFGS-B 
        
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                   'eps': 1e-10, 
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})
        # Adam 
        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
         
        #loggers:
        self.loss_total = []
        self.loss_res_log = []
        self.loss_ic_R_log = []
        self.loss_ic_C_log = []
        self.loss_ub_R_log = []
        self.loss_lb_R_log = []
        self.loss_ub_C_log = []
        self.loss_lb_C_log = []
      
        
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()
        
    def  get_collocations(self, soil, n):
        z = np.random.uniform(soil[0], soil[1], n).reshape(-1, 1)
        t =  np.random.uniform(soil[2], soil[3], n).reshape(-1, 1)
        return t, z      

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
    
    
    def PINNs(self, X, weights, biases, A):
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
                    H = tf.tanh(20 * A[l]*H)
                else:
                    H = tf.tanh(H)
        return H


    def net_res(self, t, z):  # PINNs
        X = tf.concat([t, z],1)

        psi, c = self.PINNs(X, self.weights_pinns, self.biases_pinns, self.A_pinns)
        psi =  self.net_psi(psi, self.weights_psi, self.biases_psi, self.A_psi)    
        c = self.net_c(c, self.weights_c, self.biases_c, self.A_c)
        
        # Van Genuchten
        theta= self.theta_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg)
        K= self.K_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg, self.ksvg)
        
        
        # Automatic DF
        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]
        q = -K*(psi_z + 1.0)  # water flux
        c_t = tf.gradients(c, t)[0]
        c_z = tf.gradients(c, z)[0]
        c_zz = tf.gradients(c_z, z)[0]
        D = self.diffusion_term(theta, q, self.thetaSvg,self.DL,self.Dw)
        D_z = tf.gradients(D, z)[0]
          
        f = theta_t - K_z*psi_z- K*psi_zz - K_z  # residual for Richards equation
        fc=   theta*c_t- c_z*D_z-D*c_zz + q*c_z # residual for cde equation
        
        return  psi, f, c, fc 
    
    def net_ic(self, t, z):  
        X = tf.concat([t, z],1)
        psi, c = self.PINNs(X, self.weights_pinns, self.biases_pinns, self.A_pinns)
        psi =  self.net_psi(psi, self.weights_psi, self.biases_psi, self.A_psi)    
        c = self.net_c(c, self.weights_c, self.biases_c, self.A_c)
        return  psi, c

    def net_up(self, t, z):  
        X = tf.concat([t, z],1)
        psi, c = self.PINNs(X, self.weights_pinns, self.biases_pinns, self.A_pinns)
        psi =  self.net_psi(psi, self.weights_psi, self.biases_psi, self.A_psi)    
        c = self.net_c(c, self.weights_c, self.biases_c, self.A_c)
        
        theta= self.theta_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg)
        K= self.K_function(psi, self.thetaRvg, self.thetaSvg, self.alphavg, self.nvg, self.mvg, self.ksvg)
        
        psi_z = tf.gradients(psi, z)[0]
        q = -K*(psi_z + 1.0)  # water flux
        c_z = tf.gradients(c, z)[0]
        D = self.diffusion_term(theta,q,self.thetaSvg,self.DL,self.Dw)
        qc=  - D*c_z + q*c      
        return  q, qc
    
    def net_dw(self, t, z): 
        X = tf.concat([t, z],1)
        psi, c = self.PINNs(X, self.weights_pinns, self.biases_pinns, self.A_pinns)
        psi =  self.net_psi(psi, self.weights_psi, self.biases_psi, self.A_psi)    
        c = self.net_c(c, self.weights_c, self.biases_c, self.A_c)   
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
    
    def diffusion_term(self, theta, q, thetas, DL, Dw):
        return DL * tf.abs(q) + tf.pow(theta, 7.0/3) * Dw / tf.pow(thetas, 2.0)


    def train(self, N_iter):
        tf_dict = {self.t_res_tf: self.t_res,
                       self.z_res_tf: self.z_res,
                       self.t_ic_tf: self.t_ic,
                       self.z_ic_tf: self.z_ic,
                       self.t_ub_tf: self.t_ub,
                       self.z_ub_tf: self.z_ub,
                       self.t_lb_tf: self.t_lb,
                       self.z_lb_tf: self.z_lb}
        
        start_time = time.time()
        # Adam
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict) 
         # prints the iteration number, loss value, and elapsed time every 10 iterations
       
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_icR_value, loss_icC_value, loss_ubR_value, loss_ubC_value, loss_lbR_value,\
                loss_lbC_value   = self.sess.run([self.loss, self.loss_res, self.loss_ic_R, self.loss_ic_C, self.loss_ub_R, self.loss_ub_C, self.loss_lb_R, self.loss_lb_C], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic_R: %.3e, Loss_ic_C: %.3e, Loss_ubR: %.3e, Loss_ubC: %.3e, Loss_lbR: %.3e, Loss_lbC: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_icR_value, loss_icC_value, loss_ubR_value, loss_ubC_value,loss_lbR_value, loss_lbC_value, elapsed))
                start_time = time.time()
            
                self.loss_total.append(loss_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_ic_R_log.append(loss_icR_value)
                self.loss_ic_C_log.append(loss_icC_value)
                self.loss_ub_R_log.append(loss_ubR_value)
                self.loss_lb_R_log.append(loss_lbR_value)
                self.loss_ub_C_log.append(loss_ubC_value)
                self.loss_lb_C_log.append(loss_lbC_value)
                
        
        # L-BFGS-B
    
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])
        
        # the final loss value is computed 
        loss_value = self.sess.run(self.loss, tf_dict)
       

    def callback(self):
        #print('Loss: %.3e' %(loss))
        pass
    
 
    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                       self.z_res_tf: z_star,
                  self.t_ub_tf: t_star,
                  self.z_ub_tf: z_star}
        psi = self.sess.run(self.psi_pred, tf_dict)                
        c = self.sess.run(self.c_pred, tf_dict)
        flux = self.sess.run(self.q_up_pred, tf_dict)
        total_loss = self.loss_total
        res_loss = self.loss_res_log
        IC_R_loss = self.loss_ic_R_log
        IC_c_loss  = self.loss_ic_C_log
        up_R_loss  =  self.loss_ub_R_log
        lb_R_loss = self.loss_lb_R_log
        up_C_loss = self.loss_ub_C_log
        lb_C_loss  =self.loss_lb_C_log
        return psi, c, flux, total_loss, res_loss, IC_R_loss, IC_c_loss, up_R_loss, lb_R_loss, up_C_loss, lb_C_loss
    
def main_loop(soil, size, num_layers_pinns, num_neurons_pinns, num_layers_mpinns, num_neurons_mpinns, number_random, it):
 
    # reset the graph and set random seeds
    tf.compat.v1.reset_default_graph() # clear all (equivalent in MATLAB)
    tf.compat.v1.set_random_seed(0) # TensorFlow's random generator fixed
    random.seed(0) # Python's random generator fixed
    np.random.seed(0) # NumPy's random generator fixed
    
   
    layers_pinns = np.concatenate([[2], num_neurons_pinns*np.ones(num_layers_pinns), [2]]).astype(int).tolist()
    layers_mpinns = np.concatenate([[1], num_neurons_mpinns*np.ones(num_layers_mpinns), [1]]).astype(int).tolist()


   # the PINNs object
    model = PINN_MPINN(soil, size, layers_pinns, layers_mpinns, LAA=True) 
    #train
    model.train(it)

    #prediction
    data = pd.read_csv(f"Hydrus1D_Data.csv")
    t = data['time'].values[:,None]
    z = data['depth'].values[:,None]
    Z_star = np.hstack((t, z))
    t_star = Z_star[:,0:1] # time
    z_star = Z_star[:,1:2] # depth
    
    psi, c, flux, total_loss, res_loss, IC_R_loss, IC_c_loss, up_R_loss, lb_R_loss, up_C_loss, lb_C_loss= model.predict(t_star, z_star)
    
    #   ### save the predicted data
    pred_data = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                         'psi_pred': psi.flatten(),
                         'c_pred': c.flatten(), 'flux_pred': flux.flatten()})

    pred_data.to_csv(f"./Results/Test1_MPINNPINNss_Adaptive_activation_predicted_data.csv")
    
    loss_data = pd.DataFrame({'total_loss': total_loss, 'res_loss': res_loss,
                         'ic_R_loss': IC_R_loss,
                         'up_R_loss': up_R_loss,
                         'lb_R_loss': lb_R_loss,
                        'ic_c_loss': IC_R_loss,
                         'up_c_loss': up_R_loss,
                         'lb_c_loss': lb_R_loss })
    loss_data.to_csv(f"./Results/PINNs_MPINNs_Adaptive_activation_loss_data.csv")


soil =[0.0,-1.0,0.0,1.0] #[zmin,zmax, Tinitial, Tfinal] 1D soil
size = [10000, 100, 1000, 100]  #nb of collocation points: [res, ic, upbc, dwbc] = [10000, 100, 1000, 100]
num_layers_pinns = [5,6,8]
num_layers_mpinns = [5,6,8]
num_neurons_pinns = [50,10,15]
num_neurons_mpinns = [50,10,15]
number_random = [111]

main_loop(soil, size, num_layers_pinns[0], num_neurons_pinns[0], num_layers_mpinns[0], num_neurons_mpinns[0], number_random[0], 20000)
    
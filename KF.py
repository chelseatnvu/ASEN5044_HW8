# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:00:14 2018

@author: Chelsea
"""
from scipy.io import loadmat
import numpy as np

hw8data = loadmat('hw8problem1_data.mat')
hw8data = hw8data["xasingle_truth"]
meas = np.empty((2,201))
meas[0,:] = hw8data[0,:]
meas[1,:] = hw8data[2,:]


nomeas=np.shape(meas)[1]
H_k=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.]])
Om=.045
dt=.5
F=np.array([[1., np.sin(Om*dt)/Om, 0., -(1.-np.cos(Om*dt))/Om],\
[0., np.cos(Om*dt), 0., -np.sin(Om*dt)],\
[0., (1.-np.cos(Om*dt))/Om, 1., np.sin(Om*dt)/Om],\
[0., np.sin(Om*dt), 0., np.cos(Om*dt)]])

R_a = np.array([[20,.05],[.05,20]])
Q_a = np.zeros((4,4))

#initialize xhato and Po
xhat_init=np.array([0,85*np.cos(np.pi/4),0,-85*np.sin(np.pi/4)])
p_init=900*np.diag([10,2,10,2])
xks=[]
twosigmas=[]
for i in range(nomeas):
    if i==0:
        x_k_plus = xhat_init
        p_k_plus = p_init
   
    #time update step
    x_k_minus = F@x_k_plus
    p_k_min = F @ p_k_plus @ F.T + Q_a
    k_k = p_k_min @ H_k.T @ np.linalg.inv(H_k @ p_k_min @ H_k.T + R_a)
    
    #measurement update step
    x_k_plus = x_k_minus + (k_k @ (meas[:,i] - H_k @ x_k_minus))
    p_k_plus = (np.identity(4) - k_k @ H_k) @ p_k_min
    twosigmas.append([2*np.sqrt(p_k_plus[0,0]),2*np.sqrt(p_k_plus[1,1]),2*np.sqrt(p_k_plus[2,2]),2*np.sqrt(p_k_plus[3,3])])
    xks.append(x_k_plus)
xks=np.array(xks)
twosigmas=np.array(twosigmas)
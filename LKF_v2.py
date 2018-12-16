# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:30:51 2018

@author: Chelsea
"""
import numpy as np
import LKF_mod as L
from scipy.io import loadmat
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


def meas(state_sc,IDs,time):
    y_n = []
    for i in IDs:
        xs,xsd,ys,ysd = L.statestation(i,time)
        x,xd,y,yd = state_sc
        rho = np.sqrt((x-xs)**2+(y-ys)**2)
        rho_d = ((x-xs)*(xd-xsd)+(y-ys)*(yd-ysd))/rho
        phi = np.arctan((y-ys)/(x-xs))
        y_n.append([rho,rho_d,phi,i])
#    y_n = np.array(y_n)
    return y_n

mu = 398600 #km^3/s^2
Re = 6.378e3  #km
we = 2*np.pi/86400  #rad/s


data = loadmat('orbitdeterm_finalproj_KFdata.mat')
R = data['Rtrue']
ydata = data['ydata'][0,:]
tvec = data['tvec'][0]
T = 14000
#Initial Conditions
ro=np.array([6678.,0.])    #km
ro_mag = np.linalg.norm(ro)
vo=np.array([0.,ro_mag*np.sqrt(mu/ro_mag**3)])    #km/s

#generate nominal state
x_star, time = L.solvetraj(ro[0],vo[0],ro[1],vo[1],T+10.)
#generate nominal measurements y
y_nom = []
for i in range(len(ydata)):
    IDs_vis = ydata[i] #stations we have measurements from
    if IDs_vis.size<1:
        y_nom.append([])
    else:
        IDs_vis = IDs_vis[3,:]
        y_nom.append(meas(x_star[i,:],IDs_vis,tvec[i]))
        
dt = 10. #s

#call matrices
gamma = L.gengamma()
G = dt * L.genB()
Q = data['Qtrue']
#initialize x0 and P0
x_init = np.array([ro[0],vo[0],ro[1],vo[1]]).reshape((4,1))
P_init = 9000 * np.eye(4)
#dy_init = np.array([0.,0.,0.])
        
xs_est=[]
ys_est = []
for i in range(1,len(ydata)):
    if ydata[i].size==0:
        dxp = dxp
        Pp = Pp
        continue
    IDs_vis = ydata[i][3,:] #stations we have measurements from
    if i==1:
        dxp = x_init
        Pp = P_init
#        dy = dy_init
        Xs,Ys,Xsd,Ysd = L.statestation(1,0.)
        H = L.genCi(x_init[0][0],x_init[2][0],x_init[1][0],x_init[3][0],Xs,Ys,Xsd,Ysd)
        K = Pp @ H.T @ R
    #calculate Jacobians at each time step
    F = np.eye(4) + dt*L.genA(x_star[i,0],x_star[i,2])
    Omega = dt*gamma
    H = L.genCstack2(x_star[i,:],IDs_vis,tvec[i])
    #time update
#    du = u - u_star
    dxm = (F @ dxp).reshape(4,1) #+ G @ du
    Pm = F @ Pp @ F.T + Omega @ Q @ Omega.T
    #build R matrix
    nomeas = len(IDs_vis)
    Rk = R
    if nomeas>1:
        for j in range(nomeas-1):
            Rk = block_diag(Rk,R)
    #measurement update
    K = Pm @ H.T @ np.linalg.inv(H @ Pm @ H.T + Rk)
    dy = ydata[i] - np.array(y_nom[i]).T
    dy = dy[0:3,:].flatten()  #cut out station ID and stack into 1 column
    dxp = dxm + K @ (dy.reshape((len(dy),1)) - H @ dxm)
    Pp = (np.eye(4) - K @ H) @ Pm
    xs_est.append(dxp+x_star[i,:].reshape(4,1))
    ys_est.append(H@dxp+np.array(y_nom[i])[:,0:3].flatten().reshape(3*nomeas,1))
xs_est = np.array(xs_est)
fig,ax = plt.subplots()
ax.plot(xs_est[:500,0,0],xs_est[:500,2,0])

fig2,ax2=plt.subplots(3,1)


for i in range(len(ys_est)):
    if ys_est[i].size==3:
        ax2[0].scatter(tvec[i],ys_est[i][0],s=2)
        ax2[1].scatter(tvec[i],ys_est[i][1],s=2)
        ax2[2].scatter(tvec[i],ys_est[i][2],s=2)
    if ys_est[i].size==6:
        ax2[0].scatter(tvec[i],ys_est[i][0],s=2)
        ax2[0].scatter(tvec[i],ys_est[i][3],s=2)
        ax2[1].scatter(tvec[i],ys_est[i][1],s=2)
        ax2[1].scatter(tvec[i],ys_est[i][4],s=2)
        ax2[2].scatter(tvec[i],ys_est[i][2],s=2)
        ax2[2].scatter(tvec[i],ys_est[i][5],s=2)
ax2[0].set_ylabel('rho')
ax2[1].set_ylabel('rho rate')
ax2[2].set_ylabel('phi')


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:06:54 2018

@author: Chelsea
"""

import LKF_mod as L
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.stats import chi2


mu = 398600 #km^3/s^2
dt = 10 #s


data = loadmat('orbitdeterm_finalproj_KFdata.mat')
Qtrue = data['Qtrue']
Q = data['Qtrue']
R = data['Rtrue']

#Initial Conditions
ro=np.array([6678.,0.])    #km
ro_mag = np.linalg.norm(ro)
vo=np.array([0.,ro_mag*np.sqrt(mu/ro_mag**3)])    #km/s



#call matrices
gamma = L.gengamma()
#initialize x0 and P0
#x_init = np.array([ro[0],vo[0],ro[1],vo[1]]).reshape((4,1))
dx_init = np.zeros(4).reshape((4,1))
#dx_init = np.array([.01,.1,.01,.1])
P_init = .005*np.diag([100,1,100,1])

#plot visible stations
'''fig2,ax2 = plt.subplots()
for i in range(len(IDs_vis)):
    if len(IDs_vis[i])==1:
        ax2.scatter(time[i],IDs_vis[i])
    if len(IDs_vis[i])==2:
        ax2.scatter(time[i],IDs_vis[i][0])
        ax2.scatter(time[i],IDs_vis[i][1])
ax2.set_title('Station IDs Visible Over Time')'''

num_MC = 1  #number of Monte Carlo runs
NEES_all = []
NIS_all = []
#start with 0 perturbation in ground truth, take out process noise
for k in range(num_MC):
    #find nominal solution
    TOF=1500
    noise = np.random.multivariate_normal([0,0],Qtrue,size=int(TOF/dt))
    x_star_nonoise = L.solvetraj(ro[0],ro[1],vo[0],vo[1],TOF)[0]   #no noise
    x_star, time = L.solvetraj2(ro[0],ro[1],vo[0],vo[1],TOF,noise)
    y_noisy, y_nom_nonoise, meas_noise, IDs_vis = L.geny_nom(R,TOF,time,x_star)
    xs_est=[]
    x_error = []
    y_error = []
    ys_est = []
    vis_stns = []
    NEES = []
    NIS = []
    sigma = []
    
    for i in range(len(time)):
        #initialization
        print('x actual',x_star[i,:])
        if i==0:
            dxp = dx_init
            Pp = P_init
#            Xs,Ys,Xsd,Ysd = L.statestation(1,0.)
#            H = L.genCi(x_init[0][0],x_init[2][0],x_init[1][0],x_init[3][0],Xs,Ys,Xsd,Ysd)
#            K = Pp @ H.T @ R
        #if empty data, append empty lists, and continue to next iteration
        if len(y_nom_nonoise[i])==0:
            dxp = dxp
            Pp = Pp
            xs_est.append(np.full(4,np.nan).reshape(4,1))
            ys_est.append(np.full(3,np.nan).reshape(3,1))
            vis_stns.append([time[i],np.nan])
            continue
        #calculate Jacobians at each time step
        F = np.eye(4) + dt*L.genA(x_star_nonoise[i,0],x_star_nonoise[i,2])
        Omega = dt*gamma
        H = L.genCstack2(x_star_nonoise[i,:],IDs_vis[i],time[i])
        #time update
    #    du = u - u_star
        dxm = (F @ dxp).reshape(4,1) #+ G @ du
        Pm = F @ Pp @ F.T + Omega @ Q @ Omega.T
        #build R matrix
        nomeas = len(IDs_vis[i])
        Rk = R
        if nomeas>1:
            for j in range(nomeas-1):
                Rk = block_diag(Rk,R)

        #measurement update
        K = Pm @ H.T @ np.linalg.inv(H @ Pm @ H.T + Rk)#
        dy = y_noisy[i].flatten().reshape((y_noisy[i].size,1)) - \
        (H@dxm + y_nom_nonoise[i].flatten().reshape((y_nom_nonoise[i].size,1)))
        dxp = dxm + K @ (dy.reshape((len(dy),1)) - H @ dxm)
        Pp = (np.eye(4) - K @ H) @ Pm
        sigma.append([np.sqrt(Pp[0,0]),np.sqrt(Pp[1,1]),np.sqrt(Pp[2,2]),\
                      np.sqrt(Pp[3,3])])
        xest = dxp+x_star_nonoise[i,:].reshape(4,1)
        xs_est.append(xest)
        x_error.append(xest - x_star[i,:].reshape((x_star[i,:].size,1)))
        #NIS/NEES calculation
        eyk = dy #diff b/w noisy measurement and H@dxm + nominal measurement
        exk = xest - x_star[i,:].reshape((x_star[i,:].size,1))
        ys_est.append([time[i],H@dxm + y_nom_nonoise[i].flatten().reshape((y_nom_nonoise[i].size,1))])
        if dy.size==3:
            y_error.append([time[i],dy.reshape(dy.size)])
        elif dy.size==6:
            y_error.append([time[i],dy[0:3]])
            y_error.append([time[i],dy[3:]])

        #split up eyks for data from 2 stations
        S = H @ Pm @ H.T + Rk
        if nomeas == 1:
            nis = eyk.reshape(1,3) @ np.linalg.inv(S) @ eyk.reshape(3,1)
        elif nomeas == 2:
            nis = eyk.reshape(1,6) @ np.linalg.inv(S) @ eyk.reshape(6,1)
        NIS.append([time[i],nis[0][0]])
#        nees = dxp.reshape(1,4) @ np.linalg.inv(Pm) @ dxp.reshape(4,1)
        nees = exk.T @ np.linalg.inv(Pm) @ exk
        NEES.append([time[i],nees[0][0]])
#        print('xest:',xest)
#        print('xest -xnom',xest.T-x_star[i,:])

#    y_error = np.array(y_error)
    sigma=np.array(sigma)
    x_error=np.array(x_error)
    NIS = np.array(NIS)
    NEES = np.array(NEES)
    xs_est = np.array(xs_est)
    vis_stns=np.array(vis_stns)
    NIS_all.append(NIS)
    NEES_all.append(NEES)
NIS_all = np.array(NIS_all)[:,:,1]
NIS_avg = np.mean(NIS_all,axis=0)
NEES_all = np.array(NEES_all)[:,:,1]
NEES_avg = np.mean(NEES_all,axis=0)

r1_nees = chi2.ppf(0.05,4*num_MC)/num_MC
r2_nees = chi2.ppf(0.95,4*num_MC)/num_MC
r1_nis = chi2.ppf(0.05,3*num_MC)/num_MC
r2_nis = chi2.ppf(0.95,3*num_MC)/num_MC

#orbit plot and x vs time plot
fig,ax=plt.subplots()
plt.suptitle('Orbits')
ax.plot(xs_est[:,0,0],xs_est[:,2,0],label='KF')
ax.plot(x_star[:,0],x_star[:,2],label='nominal')
ax.legend(loc='best')
ax.set_ylabel('y')
ax.set_xlabel('x')

fig6,ax6=plt.subplots(2,2)
plt.suptitle('States Over Time')
ax6[0,0].plot(time,xs_est[:,0,0],label='KF')
ax6[0,0].plot(time,x_star[:,0],label='nom.')
ax6[0,0].legend(loc='best')
ax6[0,0].set_ylabel('x')
ax6[0,0].set_xlabel('t')
ax6[0,1].plot(time,xs_est[:,1,0],label='KF')
ax6[0,1].plot(time,x_star[:,1],label='nom.')
ax6[0,1].legend(loc='best')
ax6[0,1].set_ylabel('x dot')
ax6[0,1].set_xlabel('t')
ax6[1,0].plot(time,xs_est[:,2,0],label='KF')
ax6[1,0].plot(time,x_star[:,2],label='nom.')
ax6[1,0].legend(loc='best')
ax6[1,0].set_ylabel('y')
ax6[1,0].set_xlabel('t')
ax6[1,1].plot(time,xs_est[:,3,0],label='KF')
ax6[1,1].plot(time,x_star[:,3],label='nom.')
ax6[1,1].legend(loc='best')
ax6[1,1].set_ylabel('y dot')
ax6[1,1].set_xlabel('t')

#x error plot
fig4,ax4=plt.subplots(2,2)
plt.suptitle('State Errors and 2 Sigma Bounds (dashed)')
ax4[0,0].scatter(time,x_error[:,0,0])
ax4[0,0].plot(time,2*sigma[:,0],'--',color='orange')
ax4[0,0].plot(time,-2*sigma[:,0],'--',color='orange')
ax4[0,0].set_xlabel('time (s)')
ax4[0,0].set_ylabel('x (m)')
ax4[0,1].scatter(time,x_error[:,1,0])
ax4[0,1].plot(time,2*sigma[:,1],'--',color='orange')
ax4[0,1].plot(time,-2*sigma[:,1],'--',color='orange')
ax4[0,1].set_xlabel('time (s)')
ax4[0,1].set_ylabel('xdot (m/s)')
ax4[1,0].scatter(time,x_error[:,2,0])
ax4[1,0].plot(time,2*sigma[:,2],'--',color='orange')
ax4[1,0].plot(time,-2*sigma[:,2],'--',color='orange')
ax4[1,0].set_xlabel('time (s)')
ax4[1,0].set_ylabel('y (m)')
ax4[1,1].scatter(time,x_error[:,3,0])
ax4[1,1].plot(time,2*sigma[:,3],'--',color='orange')
ax4[1,1].plot(time,-2*sigma[:,3],'--',color='orange')
ax4[1,1].set_xlabel('time (s)')
ax4[1,1].set_ylabel('ydot (m/s)')

#y error plot
fig5,ax5=plt.subplots(3,1)
plt.suptitle('Measurement Errors')
ts = [item[0] for item in y_error]
ax5[0].scatter(ts,[item[1][0] for item in y_error])
ax5[0].set_xlabel('time (s)')
ax5[0].set_ylabel('range (m)')
ax5[1].scatter(ts,[item[1][1] for item in y_error])
ax5[1].set_xlabel('time (s)')
ax5[1].set_ylabel('range rate (m/s)')
ax5[2].scatter(ts,[item[1][2] for item in y_error])
ax5[2].set_xlabel('time (s)')
ax5[2].set_ylabel('phi (rad.)')

#y estimate plot
fig7,ax7=plt.subplots(3,1)
plt.suptitle('Y estimate')
ts = [item[0] for item in ys_est]
ax7[0].scatter(ts,[item[1][0][0] for item in ys_est])
ax7[1].scatter(ts,[item[1][1][0] for item in ys_est])
ax7[2].scatter(ts,[item[1][2][0] for item in ys_est])

#NIS/NEES plot
fig3,ax3=plt.subplots(2,1)
ax3[0].scatter(time[27:],NIS_avg[27:])
ax3[0].plot([time[0],time[-1]],[r1_nis,r1_nis],label='r1',color='red')
ax3[0].plot([time[0],time[-1]],[r2_nis,r2_nis],label='r2',color='green')
ax3[0].set_ylabel('NIS')
ax3[0].set_xlabel('Time (s)')
ax3[0].legend(loc='best')
#ax3[1].scatter(time[10:],NEES_avg[10:])
ax3[1].scatter(time,NEES_avg)
ax3[1].plot([time[0],time[-1]],[r1_nees,r1_nees],label='r1',color='red')
ax3[1].plot([time[0],time[-1]],[r2_nees,r2_nees],label='r2',color='green')
ax3[1].legend(loc='best')
ax3[1].set_ylabel('NEES')
ax3[1].set_xlabel('Time (s)')


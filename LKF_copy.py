#LKF straight from my EKF

import numpy as np
from scipy.io import loadmat
import scipy as sp
import scipy.linalg as LA
from matplotlib import pyplot as plt
from funcs import eom,A,C,y,gen_meas
from scipy.stats import chi2
import sys

#Load instructor-provided values
locals().update(loadmat('orbitdeterm_finalproj_KFdata'))
#Qtrue = np.diag([1e-7,1e-7])
y_data = []
for a in ydata[0][1:]:
    if np.size(a) == 8:
        a = np.append(a[:,0],a[:,1],axis=0).reshape(8,1)
    y_data.append(a)
y_data = y_data
    
#Constants
mu = 3.986004415e5

pert = np.array([0.1,0.001,0.1,0.001])

NEES = []
NIS = []
NEESbar = []
NISbar = []
sampSize = 10
for runs in range(sampSize): # -------------------------------------------------------------------
    #Initial Conditions and Solver Arguments
    Qtrue = 1*Qtrue
    Rtrue = 1*Rtrue
    dist = 6678
    x0 = np.array([dist,0,0,dist*np.sqrt(mu/dist**3)])
    tof = 14000
    step = 10
    time_range = np.arange(0,tof+step,step)
    gamma = np.array([[0, 0],[1, 0],[0, 0],[0, 1]])
    O = step*gamma
    dirt = np.random.multivariate_normal([0,0],Qtrue,size=int(tof/step)+1000)
    not_dirt = np.zeros((len(dirt),2))
    
    #Calculate the truth value and nominal x_star ------------------------------------------------
    x_star = sp.integrate.odeint(eom,x0,time_range,args=(mu,not_dirt,step))
    x_real = sp.integrate.odeint(eom,x0,time_range,args=(mu,dirt,step))
    
    #Measurements --------------------------------------------------------------------------------
    Rtrue = 1*np.diag([0.1,1,0.1])
    y_data = gen_meas(x0,tof,step,mu,x_real,Rtrue)
    
    #Extended Kalman Filter ----------------------------------------------------------------------    
    
    #Simulation
    x_ekf = []
    y_ekf = []
    P_ekf = []
    e_ekf = []
    y_lin = []
    eps_x = []
    eps_y = []
    K_ekf = []
    ping_list = []
    y_save = []
    y1 = []
    y2 = []
    for t in np.arange(0,tof+step,step):
        k = int(t/step)
        if t==0:
            dx_p = pert
            P_p = 1*np.diag([1,0.001,1,0.001])
            Rkf = Rtrue*1.2
            Q = Qtrue*1.2
#            Q = np.zeros(Q.shape)
        else:
            #Linearize A
            A_mat = A(x_star[k-1],mu)
            F = np.eye(4) + step*A_mat + 0.5*step**2 * A_mat @ A_mat
            
            #Update Step
            P_m = 1*(F @ P_p @ F.T + O @ Q @ O.T)
            dx_m = F @ dx_p
            
            #Calculate some y guesses based on which measurements we have
            meas = y_data[k]
            if not meas.size:
                pings = 0,
                e = []
            else:
                meas_list = np.split(meas,len(meas)/4)
                meas, pings = [],[]
                for item in meas_list:
                    meas.append(item[:-1])
                    pings.append(item[-1])
                meas = np.array(meas)
                meas = meas.reshape(np.size(meas),1)
                #Calculate H
                H = C(x_star[k,:],mu,t,pings)
                #Calculate nominal sensor measurement
                y_val = y(x_star[k,:],mu,t,pings)
                y_save.append(y_val)
                y_m = []
                for item in np.split(y_val,len(y_val)/5):
                    y_ekf.append(item)
                    y_m.append(item[:-2])
                    y1.append(item)
                y_m = np.array(y_m)
                y_m = y_m.reshape(np.size(y_m),1)
                for i in range(int(len(H)/3)):
                    y2.append(np.append(H[3*i:3*i+3,:]@dx_m.reshape(4,1),np.zeros((2,1))))
                dy = meas - y_m
                for item in np.split(dy,len(dy)/3):
                    e_ekf.append(item)
            
            #If measurements exist, perform correction step, otherwise skip it
            if not len(dy):
                dx_p = dx_m
                P_p = P_m
                eps_x.append(dx_m.T @ LA.inv(P_p) @ dx_m)
                eps_y.append(np.array([[0]]))
            else:
                R = np.kron(np.eye(int(np.size(dy)/3)),Rkf)
                K = P_m @ H.T @ LA.inv(H @ P_m @ H.T + R)
                K_ekf.append(K)
                S_k = H @ P_m @ H.T + R
                dy_m = H @ dx_m.reshape(4,1)
                dx_p = dx_m.reshape(4,1) +  K @ (dy - dy_m)
                e_x = x_real[k,:].reshape(4,1) - dx_p.reshape(4,1) - x_star[k,:].reshape(4,1)
                e_y = meas - y_m - dy_m
                for i in range(int(len(e_y)/3)):
                    e_y[3*i-1] = np.arctan2(np.sin(meas[3*i-1]-(y_m+dy_m)[3*i-1]),np.cos(meas[3*i-1]-(y_m+dy_m)[3*i-1]))
                P_p = (np.eye(4) - K @ H ) @ P_m
                eps_x_val = e_x.T @ LA.inv(P_p) @ e_x
                eps_x.append(eps_x_val)
                eps_y_val = e_y.T @ LA.inv(S_k) @ e_y
                if eps_x_val > 30:
                    print('It happened', t, eps_x_val, runs)
                    print(e_x.T)
                    print(e_x.reshape(4)/np.sqrt(np.diag(P_p)))
                    print(x_real[k])
                    print(x_star[k].reshape(1,4)+dx_p.reshape(1,4))
                    print(x_star[k])
                    print(dx_p.T)
                    print(np.sqrt(np.diag(P_p)))
                    sys.exit()
                eps_y.append(eps_y_val)
                
        P_ekf.append(P_p)
        x_ekf.append(dx_p.reshape(4,1)+x_star[k,:].reshape(4,1))
    
    for i in range(len(y1)):
        y_lin.append(y1[i].reshape(5,1) + y2[i].reshape(5,1))
            
    x_ekf = np.array(x_ekf).reshape(len(x_ekf),4).T
    y_ekf = np.array(y_ekf)
    y_lin = np.array(y_lin)
    P_ekf = np.array(P_ekf)
    e_ekf = np.array(e_ekf).reshape(len(e_ekf),3)
    eps_x = np.array(eps_x).reshape(len(eps_x))
    eps_y = np.array(eps_y).reshape(len(eps_y))
    
    #NEES and NIS
    NEES.append(eps_x)
    NIS.append(eps_y)
    
#NEES, NIS ---------------------------------------------------------------------------------------
NEES = np.array(NEES).reshape(k,sampSize)
NIS = np.array(NIS).reshape(k,sampSize)
NEES_bar = np.mean(NEES,axis=1)
NIS_bar = np.mean(NIS,axis=1)
r1_NEES = chi2.ppf(0.01,4*sampSize)/sampSize
r2_NEES = chi2.ppf(0.99,4*sampSize)/sampSize
r1_NIS = chi2.ppf(0.01,3*sampSize)/sampSize
r2_NIS = chi2.ppf(0.99,3*sampSize)/sampSize
    
#Plotting ----------------------------------------------------------------------------------------
only_plot = 3,7,6,
#Spatial
if 1 in only_plot:
    fig1,ax1 = plt.subplots(1,1)
    ax1.plot(x_real[:,0],x_real[:,2],label='True')
    ax1.plot(x_ekf[0,:],x_ekf[2,:],label='Kalman')
    ax1.legend()
    plt.show()

#Truth Value vs. Time
if 2 in only_plot:
    fgi2,ax2 = plt.subplots(2,2)
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax2[i,j].plot(time_range,x_real[:,k], label='True')
            ax2[i,j].plot(time_range,x_ekf[k,:], label='Kalman')
            ax2[i,j].legend()
    plt.show()
    
#Y Measurements vs Time
if 3 in only_plot:
    fig3, ax3 = plt.subplots(2,2)
    ylabels = ['Rho','Rho_D','Phi','Station ID']
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax3[i,j].plot(y_ekf[:,4,0],y_ekf[:,k,0], '.')
            ax3[i,j].set_ylabel(ylabels[k])
            ax3[i,j].set_xlabel('Time')
    plt.suptitle('Y Values vs Time')
    plt.show()
    
#Error vs. Time
if 4 in only_plot:
    fig4, ax4 = plt.subplots(2,2)
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax4[i,j].plot(time_range,np.abs(x_ekf[k,:]-x_real[:,k]))
    plt.show()
    
#Covariance vs Time
if 5 in only_plot:
    fig5, ax5 = plt.subplots(4,4)
    for i in 0,1,2,3:
        for j in 0,1,2,3:
            k = 4*i+j
            ax5[i,j].plot(time_range[:],P_ekf[:,i,j])
    plt.show()
            
#NEES,NIS Test
if 6 in only_plot:
    fig6, ax6 = plt.subplots(1,2)
    ax6[0].plot(NEES_bar,'.')
    ax6[0].plot(r1_NEES*np.ones(len(NEES_bar)))
    ax6[0].plot(r2_NEES*np.ones(len(NEES_bar)))
    ax6[0].set_title('NEES')
    ax6[0].set_ylabel('Chi Squared Statistic')
    ax6[1].plot(NIS_bar,'.')
    ax6[1].plot(r1_NIS*np.ones(len(NIS_bar)))
    ax6[1].plot(r2_NIS*np.ones(len(NIS_bar)))
    ax6[1].set_title('NIS')
    ax6[1].set_ylabel('Chi Squared Statistic')
    plt.show()
    
#Approximate Y Measurements vs Time
if 7 in only_plot:
    fig7, ax7 = plt.subplots(2,2)
    ylabels = ['Rho','Rho_D','Phi','Station ID']
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax7[i,j].plot(y_lin[:,4],y_lin[:,k], '.')
            ax7[i,j].set_ylabel(ylabels[k])
            ax7[i,j].set_xlabel('Time')
    plt.suptitle('Linearized Approximate Y Values vs Time')
    plt.show()
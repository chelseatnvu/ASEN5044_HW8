import numpy as np
from scipy.io import loadmat
import scipy as sp
import scipy.linalg as LA
from matplotlib import pyplot as plt
from funcs import eom,A,C,y,gen_meas
from scipy.stats import chi2

#Load instructor-provided values
locals().update(loadmat('orbitdeterm_finalproj_KFdata'))
Q = 1*Qtrue
y_data = []
for a in ydata[0][1:]:
    if np.size(a) == 8:
        a = np.append(a[:,0],a[:,1],axis=0).reshape(8,1)
    y_data.append(a)
y_data = y_data
    
#Constants
mu = 3.986004415e5

NEES = []
NIS = []
NEESbar = []
NISbar = []
sampSize = 5
for runs in range(sampSize):
    #Initial Conditions and Solver Arguments
    x0 = np.array([6678,0,0,6678*np.sqrt(mu/6678**3)])
    tof = 3000
    step = 1
    time_range = np.arange(0,tof+step,step)
    dirt = np.random.multivariate_normal([0,0],Qtrue,size=int(tof/step)+2)
    
    #My Generated Data
    Rtrue = 1*np.diag([0.1,1,0.1])
    y_data = gen_meas(x0,tof,step,mu,dirt,Rtrue)
    
    #Calculate the truth value -------------------------------------------------------------------
    x_star = sp.integrate.odeint(eom,x0,time_range,args=(mu,dirt,step))
    
    #Extended Kalman Filter ----------------------------------------------------------------------
    #Time-Invariant Linearization Matrices
    gamma = np.array([[0, 0],[1, 0],[0, 0],[0, 1]])
    O = step*gamma
    
    #Simulation
    x_ekf = []
    y_ekf = []
    P_ekf = []
    e_ekf = []
    y_lin = []
    eps_x = []
    eps_y = []
    ping_list = []
    for t in np.arange(0,tof+step,step):
        k = int(t/step)
        if t==0:
            x_hat_p = x0
            P_hat_p = 2*np.diag([0.1286,2.958e-6,0.064,2.617e-6])
            Rkf = 1*Rtrue
        else:
            #Integrate x forward to current time step with odeint
            x_hat_m = sp.integrate.odeint(eom,x_hat_p.reshape(4,),[t-step,t],args=(mu,dirt,step))[-1,:]
            
            #Linearize F to determine covariance
            F = np.eye(4) + step*A(x_hat_p,mu)
            P_hat_m = F @ P_hat_p @ F.T + O @ Q @ O.T
            
            #Calculate some y guesses based on which measurements we have
            meas = y_data[k]
            if not meas.size:
                pings = 0,
                e = []
            elif np.size(meas) == 4:
                meas, pings = meas[:-1],meas[-1]
                H = C(x_hat_m,mu,t,pings)
                y_hat_m = y(x_hat_m,mu,t,pings)
                y_ekf.append(y_hat_m.reshape(5,1))
                y_hat_m = y_hat_m[:-2]
                lin_appr = np.append(np.append(H @ x_hat_m,pings),t)
                y_lin.append(lin_appr)
                e = meas - y_hat_m
                e_ekf.append(e)
            elif np.size(meas) == 8:
                meas, pings = np.append(meas[0:3],meas[4:-1], axis=0), [*meas[3],*meas[-1]]
                H = C(x_hat_m,mu,t,pings)
                y_hat_m = y(x_hat_m,mu,t,pings)
                y_ekf.append(y_hat_m[:5])
                y_ekf.append(y_hat_m[5:])
                y_hat_m = y_hat_m[[0,1,2,5,6,7]]
                lin_appr = np.append(np.append(H[:3,:] @ x_hat_m,pings[0]),t)
                y_lin.append(lin_appr)
                lin_appr = np.append(np.append(H[3:,:] @ x_hat_m,pings[1]),t)
                y_lin.append(lin_appr)
                e = meas[0:3] - y_hat_m[0:3]
                e_ekf.append(e)
                e = meas[3:] - y_hat_m[5:8]
                e_ekf.append(e)
                e = meas - y_hat_m
            else:
                print('Error')
            
            #If measurements exist, perform correction step, otherwise skip it
            if not len(e):
                x_hat_p = x_hat_m
                P_hat_p = P_hat_p
                e_x = x_star[k].reshape(4,1) - x_hat_p.reshape(4,1)
                eps_x.append(e_x.T @ LA.inv(P_hat_p) @ e_x)
                eps_y.append(np.array([[0]]))
            else:
                R = np.kron(np.eye(int(np.size(e)/3)),Rkf)
                K = P_hat_m @ H.T @ LA.inv(H @ P_hat_m @ H.T + R)
                S_k = H @ P_hat_m @ H.T + R
                x_hat_p = x_hat_m.reshape(4,1) + K @ e
                e_x = x_star[k].reshape(4,1) - x_hat_p.reshape(4,1)
                P_hat_p = (np.eye(4) - K @ H ) @ P_hat_m
                eps_x.append(e_x.T @ LA.inv(P_hat_p) @ e_x)
                eps_y.append(e.T @ LA.inv(S_k) @ e)
                
            P_ekf.append(P_hat_p)
        x_ekf.append(x_hat_p.reshape(4,1))
            
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
r1_NEES = chi2.ppf(0.05,4*sampSize)/sampSize
r2_NEES = chi2.ppf(0.95,4*sampSize)/sampSize
r1_NIS = chi2.ppf(0.05,3*sampSize)/sampSize
r2_NIS = chi2.ppf(0.95,3*sampSize)/sampSize
    
#Plotting ----------------------------------------------------------------------------------------
only_plot = 6,
#Spatial
if 1 in only_plot:
    fig1,ax1 = plt.subplots(1,1)
    ax1.plot(x_star[:,0],x_star[:,2],label='True')
    ax1.plot(x_ekf[0,:],x_ekf[2,:],label='Kalman')
    ax1.legend()
    plt.show()

#Truth Value vs. Time
if 2 in only_plot:
    fgi2,ax2 = plt.subplots(2,2)
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax2[i,j].plot(time_range,x_star[:,k], label='True')
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
            ax4[i,j].plot(time_range,x_ekf[k,:]-x_star[:,k])
    plt.show()
    
#Covariance vs Time
if 5 in only_plot:
    fig5, ax5 = plt.subplots(4,4)
    for i in 0,1,2,3:
        for j in 0,1,2,3:
            k = 4*i+j
            ax5[i,j].plot(time_range[1:],P_ekf[:,i,j])
    plt.show()
            
#NEES,NIS Test
if 6 in only_plot:
    fig6, ax6 = plt.subplots(1,2)
    ax6[0].plot(NEES_bar,'.')
    ax6[0].plot(r1_NEES*np.ones(len(NEES_bar)))
    ax6[0].plot(r2_NEES*np.ones(len(NEES_bar)))
    ax6[1].plot(NIS_bar,'.')
    ax6[1].plot(r1_NIS*np.ones(len(NIS_bar)))
    ax6[1].plot(r2_NIS*np.ones(len(NIS_bar)))
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
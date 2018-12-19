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
for a in ydata[0][:]:
    if np.size(a) == 8:
        a = np.append(a[:,0],a[:,1],axis=0).reshape(8,1)
    y_data.append(a)
y_data = y_data
    
#Constants
mu = 3.986004415e5

pert = np.array([-4.9766,0.1,18.143,-0.02])

NEES = []
NIS = []
NEESbar = []
NISbar = []
sampSize = 5
for runs in range(sampSize): # -------------------------------------------------------------------
    #Initial Conditions and Solver Arguments
    dist = 6678
    x0 = np.array([dist,0,0,dist*np.sqrt(mu/dist**3)])
    tof = 14000
    step = 10
    time_range = np.arange(0,tof+step,step)
    dirt = np.random.multivariate_normal([0,0],Qtrue,size=int(tof/step)+1000)
    not_dirt = np.zeros((len(dirt),2))
    
    #Calculate the truth value -------------------------------------------------------------------
    x_star = sp.integrate.odeint(eom,x0,time_range,args=(mu,not_dirt,step))
    
    #Measurements --------------------------------------------------------------------------------
    Rtrue = 1*np.diag([0.1,1,0.1])
    y_data = gen_meas(x0,tof,step,mu,x_star,Rtrue)
    
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
            x_hat_p = x0+pert
            P_hat_p = .1*np.diag([50,0.002,50,0.002])
            Rkf = Rtrue*1.1
            Q = Qtrue*1
        else:
            #Integrate x forward to current time step with odeint
            x_hat_m = sp.integrate.odeint(eom,x_hat_p.reshape(4,),[t-step,t],args=(mu,not_dirt,step))[-1,:]
            
            #Linearize F to determine covariance
            F = np.eye(4) + step*A(x_hat_p,mu)
            P_hat_m = F @ P_hat_p @ F.T + O @ Q @ O.T
            P_hat_m = 1 * P_hat_m
            
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
                H = C(x_hat_m,mu,t,pings)
                y_val = y(x_hat_m,mu,t,pings)
                y_hat_m = []
                for item in np.split(y_val,len(y_val)/5):
                    y_ekf.append(item)
                    y_hat_m.append(item[:-2])
                i = 0
                for item in np.split(H,len(y_val)/5,axis=0):
                    y_lin.append(np.append(np.append(item @ x_hat_m,pings[i]),t))
                    i += 1
                y_hat_m = np.array(y_hat_m)
                y_hat_m =y_hat_m.reshape(np.size(y_hat_m),1)
                e = meas - y_hat_m
                for item in np.split(e,len(e)/3):
                    e_ekf.append(item)
            
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
                eps_x_val = e_x.T @ LA.inv(P_hat_p) @ e_x
                eps_y_val = e.T @ LA.inv(S_k) @ e
                eps_x.append(eps_x_val)
                eps_y.append(eps_y_val)
#                if eps_x_val > 30:
#                    print('It happened', t, eps_x_val, runs)
#                    print(e_x.T)
#                    print(e_x.reshape(4)/np.sqrt(np.diag(P_hat_p)))
#                    print(x_star[k])
#                    print(x_hat_p.reshape(1,4))
#                    print(np.sqrt(np.diag(P_hat_p)))
#                    sys.exit()
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
r1_NEES = chi2.ppf(0.01,4*sampSize)/sampSize
r2_NEES = chi2.ppf(0.99,4*sampSize)/sampSize
r1_NIS = chi2.ppf(0.01,3*sampSize)/sampSize
r2_NIS = chi2.ppf(0.99,3*sampSize)/sampSize
    
#Plotting ----------------------------------------------------------------------------------------
only_plot = range(9)
#Spatial
if 1 in only_plot:
    fig1,ax1 = plt.subplots(1,1)
    ax1.plot(x_star[:,0],x_star[:,2],label='True')
    ax1.plot(x_ekf[0,:],x_ekf[2,:],label='Kalman')
    ax1.legend()
    plt.suptitle('EKF Spatial Plot')
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
    plt.suptitle('EKF States and Actual States vs. Time')
    plt.show()
    
#Y Measurements vs Time
if 93 in only_plot:
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
            ax4[i,j].plot(time_range[1:],2*np.sqrt(P_ekf[:,k,k]))
            ax4[i,j].plot(time_range[1:],-2*np.sqrt(P_ekf[:,k,k]))
    plt.suptitle('EKF State Errors vs Time')
    plt.show()
    
#Covariance vs Time
if 5 in only_plot:
    fig5, ax5 = plt.subplots(4,4)
    for i in 0,1,2,3:
        for j in 0,1,2,3:
            k = 4*i+j
            ax5[i,j].plot(time_range[1:],P_ekf[:,i,j])
    plt.suptitle('EKF Co-Variance Elements vs Time')
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
    plt.suptitle('EKF Chi Squared Test')
    plt.show()
    
#Approximate Y Measurements vs Time
if 97 in only_plot:
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
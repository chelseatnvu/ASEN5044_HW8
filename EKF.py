import numpy as np
from scipy.io import loadmat
import scipy as sp
import scipy.linalg as LA
from matplotlib import pyplot as plt
from funcs import eom,A,C,y
from scipy.stats.mstats import chisquare

#Load instructor-provided values
locals().update(loadmat('orbitdeterm_finalproj_KFdata'))
Q = Qtrue
y_data = []
for a in ydata[0]:
    if np.size(a) == 8:
        a = np.append(a[:,0],a[:,1],axis=0).reshape(8,1)
    y_data.append(a)
    
#Constants
mu = 3.986004415e5

#Initial Conditions and Solver Arguments
x0 = np.array([6678,0,0,6678*np.sqrt(mu/6678**3)])
tof =1200
step = 10
time_range = np.arange(0,tof+step,step)

#Calculate the truth value -----------------------------------------------------------------------
x_star = sp.integrate.odeint(eom,x0,time_range,args=(mu,))

#Extended Kalman Filter --------------------------------------------------------------------------
#Time-Invariant Linearization Matrices
gamma = np.array([[0, 0],[1, 0],[0, 0],[0, 1]])
O = step*gamma

#Simulation
x_ekf = []
y_ekf = []
eps_x_ekf = []
eps_y_ekf = []
for t in np.arange(0,tof+step,step):
    k = int(t/step)
    if t==0:
        x_hat_p = x0
        P_hat_p = 1*np.diag([10,0.1,10,0.1])
    else:
        #Integrate x forward to current time step with odeint
        x_hat_m = sp.integrate.odeint(eom,x_hat_p.reshape(4,),[t-step,t],args=(mu,))[-1,:]
        
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
            y_hat_m = y(x_hat_m,mu,t,pings)[:-1]
            e = meas - y_hat_m
        elif np.size(meas) == 8:
            meas, pings = np.append(meas[0:3],meas[4:-1], axis=0), [meas[3],meas[-1]]
            y_hat_m = np.append(y(x_hat_m,mu,t,pings)[0:3],y(x_hat_m,mu,t,pings)[4:-1],axis=0)
            e = meas - y_hat_m
        else:
            print('Error')
        
        #If measurements exist, perform correction step, otherwise skip it
        if not len(e):
            x_hat_p = x_hat_m
            P_hat_p = P_hat_p
        else:
            H = C(x_hat_m,mu,t,pings)
            R = np.kron(np.eye(int(np.size(e)/3)),Rtrue)
#            K = P_hat_m @ H.T @ LA.inv(H @ P_hat_m @ H.T + R)
            P_inf_m = LA.solve_discrete_are(F,H.T,O @ Q @ O.T,R)
            K = P_inf_m @ H.T @ LA.inv(H @ P_inf_m @ H.T + R)
            S_k = H @ P_hat_m @ H.T + R
            eps_y_ekf.append(e.T @ LA.inv(S_k) @ e)
            x_hat_p = x_hat_m.reshape(4,1) + K @ e
            P_hat_p = (np.eye(4) - K @ H ) @ P_hat_m
        y_ekf.append(y_hat_m.reshape(np.size(y_hat_m),1))
    eps_x_ekf.append((x_star[k,:].reshape(4,1)-x_hat_p.reshape(4,1)).T @ LA.inv(P_hat_p) @ (x_star[k,:].reshape(4,1)-x_hat_p.reshape(4,1)))
    x_ekf.append(x_hat_p.reshape(4,1))
        
x_ekf = np.array(x_ekf)
y_ekf = np.array(y_ekf)
eps_y_ekf = np.array(eps_y_ekf)
eps_x_ekf = np.array(eps_x_ekf) 

#NEES and NIS
N = len(eps_x_ekf)
x_hist, x_bins = np.histogram(N*eps_x_ekf)
deg_x, p_x = chisquare(x_hist)

#Plotting ----------------------------------------------------------------------------------------
only_plot = 3,4,5

#Truth Value Spatial
if 1 in only_plot:
    fig1,ax1 = plt.subplots(1,1)
    ax1.plot(x_star[:,0],x_star[:,2])
    plt.show()

#Truth Value vs. Time
if 2 in only_plot:
    fgi2,ax2 = plt.subplots(2,2)
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax2[i,j].plot(time_range,x_star[:,k])
    plt.show()

#EKF Spatial
if 3 in only_plot:
    fig3,ax3 = plt.subplots(1,1)
    ax3.plot(x_ekf[:,0,0],x_ekf[:,2,0])
    plt.show()
    
#Error vs. Time
if 4 in only_plot:
    fig4, ax4 = plt.subplots(2,2)
    for i in 0,1:
        for j in 0,1:
            k=2*i+j
            ax4[i,j].plot(time_range,x_ekf[:,k,0]-x_star[:,k])
    plt.show()
    
#NEES,NIS Test
if 5 in only_plot:
    fig5, ax5 = plt.subplots(1,2)
    ax5[0].hist(eps_x_ekf[:,0,0],10)
    ax5[1].hist(eps_y_ekf[:,0,0],10)
    plt.show()
import numpy as np
from scipy import linalg as LA
from numpy import linalg as LA2
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

x_val = loadmat('hw8problem1_data')['xasingle_truth']

#Part A
O = 0.045
q_w = 10
dt = 0.5
w = q_w * np.array([[2,       0.05    ],
                    [0.05,    0.5     ]])
gamma = np.array([  [0,0],
                    [1,0],
                    [0,0],
                    [0,1]   ])
A_a = np.array([[0,   1,  0,  0   ],
                [0,   0,  0,  -O  ],
                [0,   0,  0,  1   ],
                [0,   O,  0,  0   ]  ])
A_b = np.array([[0,   1,  0,  0   ],
                [0,   0,  0,  O   ],
                [0,   0,  0,  1   ],
                [0,   -O, 0,  0   ]  ])
Z = np.append(-A_a,gamma@w@gamma.T,axis=1)
Z = dt*np.append(Z,np.append(np.zeros((4,4)),A_a.T,axis=1),axis=0)
expZ = LA.expm(Z)
F_a = expZ[4:,4:].T
Q_a = F_a@expZ[0:4,4:]

Z = np.append(-A_b,gamma@w@gamma.T,axis=1)
Z = dt*np.append(Z,np.append(np.zeros((4,4)),A_b.T,axis=1),axis=0)
expZ = LA.expm(Z)
F_b = expZ[4:,4:].T
Q_b = F_b@expZ[0:4,4:]

#Part B
H = np.array([  [1,0,0,0],
                [0,0,1,0]  ])
R_a = np.array([[20,    0.05],
                [0.05,  20  ]  ])
np.random.seed(0)
y = (H@x_val[:,0] + multivariate_normal(mean=None,cov=R_a).rvs()).reshape((2,1))
for i in range(1,201):
    np.random.seed(i)
    y = np.append(y,(H@x_val[:,i] + multivariate_normal(mean=None,cov=R_a).rvs()).reshape((2,1)),axis=1)

#initialize xhato and Po
xhat_init=np.array([0,85*np.cos(np.pi/4),0,-85*np.sin(np.pi/4)])
p_init=900*np.diag([10,2,10,2])
xks=[]
twosigmas=[]
for i in range(201):
    if i==0:
        x_k_plus = xhat_init
        p_k_plus = p_init
   
    #time update step
    x_k_minus = F_a@x_k_plus
    p_k_min = F_a @ p_k_plus @ F_a.T + Q_a
    k_k = p_k_min @ H.T @ np.linalg.inv(H @ p_k_min @ H.T + R_a)
    
    #measurement update step
    x_k_plus = x_k_minus + (k_k @ (y[:,i] - H @ x_k_minus))
    p_k_plus = (np.identity(4) - k_k @ H) @ p_k_min
    twosigmas.append([2*np.sqrt(p_k_plus[0,0]),2*np.sqrt(p_k_plus[1,1]),2*np.sqrt(p_k_plus[2,2]),2*np.sqrt(p_k_plus[3,3])])
    xks.append(x_k_plus)
xks=np.array(xks).T
twosigmas=np.array(twosigmas).T

#Plotting
fig, ax = plt.subplots(1,2)

for i in (0,1):
    ax[i].plot(np.arange(1,42),y[i-1,0:41],'.',label='Position Input')
    ax[i].plot(np.arange(1,42),xks[2*i-2,0:41],label='Position (Kalman Filter)')
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('Position (m)')
    ax[i].legend()
ax[0].set_title('Easterly Position vs. Time')
ax[1].set_title('Northerly Position vs. Time')
plt.show()

#calculate estimated state error
est_st_err = np.abs(x_val - xks)

#plot estimated state error
ksplt=np.linspace(0,200,201)
fig2,ax2 = plt.subplots(2,2)
for i in range(len(ax2)):
    for j in range(len(ax2[i])):
        ax2[i,j].scatter(ksplt,est_st_err[2*i+j,:],s=2,label='Error')
        ax2[i,j].plot(ksplt,twosigmas[2*i+j,:],'g',label='Est Error Bounds')
        ax2[i,j].set_xlabel('Time (s)')
        ax2[i,j].legend()
        if (2*1+j)%2 == 0:
            ax2[i,j].set_ylabel('Position Error (m)')
        else:
            ax2[i,j].set_ylabel('Velocity Error (m/s)')
ax2[0,0].set_title('East Position')
ax2[0,1].set_title('East Velocity')
ax2[1,0].set_title('North Position')
ax2[1,1].set_title('North Velocity')
plt.tight_layout(rect=[0.1,0,0.9,1])
plt.show()

#part c
#read in A and B data
x_val_A = loadmat('hw8problem1_data')['xadouble_truth']
x_val_B = loadmat('hw8problem1_data')['xbdouble_truth']

#simulate noisy A measurements and noisy A-B measurements
y_A = (H@x_val_A[:,0] + \
       multivariate_normal(mean=None,cov=R_a).rvs()).reshape((2,1))
R_d = np.array([[10,    0.15],
                [0.15,  10  ]  ])
y_D = (H@x_val_A[:,0] - H@x_val_B[:,0] + \
       multivariate_normal(mean=None,cov=R_d).rvs()).reshape((2,1))
for i in range(1,201):
    np.random.seed(i)
    y_A = np.append(y_A,(H@x_val_A[:,i] + \
                 multivariate_normal(mean=None,cov=R_a).rvs()).reshape((2,1)),axis=1)
    y_D = np.append(y_D,(H@x_val_A[:,i] - H@x_val_B[:,i] + \
                 multivariate_normal(mean=None,cov=R_d).rvs()).reshape((2,1)),axis=1)
#stack measurements
y_s = np.concatenate((y_A,y_D),axis=0)

#define new augmented state matrices (p->prime)-->check with Connor
Fp = block_diag(F_a,F_b)
Ap = block_diag(A_a,A_b)
gammap = block_diag(gamma,gamma)
wp = block_diag(w,w)
Hp = np.array([[1, 0, 0, 0, 0, 0, 0, 0],\
               [0, 0, 1, 0, 0, 0, 0, 0],\
               [1, 0, 0, 0, -1, 0, 0, 0],\
               [0, 0, 1, 0, 0, 0, -1, 0]])
Rp = block_diag(R_a, R_d)   #especially not sure about this one... -ch

Zp = np.append(-Ap,gammap@wp@gammap.T,axis=1)
Zp = dt*np.append(Zp,np.append(np.zeros((8,8)),Ap.T,axis=1),axis=0)
expZp = LA.expm(Zp)
Fp = expZp[8:,8:].T
Qp = Fp@expZp[0:8,8:]

#initialize xhato and Po ->check with connor on this too
xa0 = [0,85*np.cos(np.pi/4),0,-85*np.sin(np.pi/4)]
xb0 = [3200,85*np.cos(np.pi/4),3200,-85*np.sin(np.pi/4)]
pa0 = [10,2,10,2]
pb0 = [11,4,11,4]
xhat_init=np.array(xa0+list(np.array(xa0)-np.array(xb0)))
p_init=900*np.diag(pa0+list(np.array(pa0)+np.array(pb0)))
xksp=[]
twosigmasp=[]
for i in range(201):
    if i==0:
        x_k_plus = xhat_init
        p_k_plus = p_init
   
    #time update step
    x_k_minus = Fp@x_k_plus
    p_k_min = Fp @ p_k_plus @ Fp.T + Qp
    k_k = p_k_min @ Hp.T @ np.linalg.inv(Hp @ p_k_min @ Hp.T + Rp)
    
    #measurement update step
    x_k_plus = x_k_minus + (k_k @ (y_s[:,i] - Hp @ x_k_minus))
    p_k_plus = (np.identity(8) - k_k @ Hp) @ p_k_min
    twosigmasp.append([2*np.sqrt(p_k_plus[0,0]),2*np.sqrt(p_k_plus[1,1]),\
                      2*np.sqrt(p_k_plus[2,2]),2*np.sqrt(p_k_plus[3,3]),\
                      2*np.sqrt(p_k_plus[4,4]),2*np.sqrt(p_k_plus[5,5]),\
                      2*np.sqrt(p_k_plus[6,6]),2*np.sqrt(p_k_plus[7,7])])
    xksp.append(x_k_plus)
xksp=np.array(xksp).T
twosigmasp=np.array(twosigmasp).T


true_state = np.concatenate((x_val_A,x_val_B),axis=0)
est_st_errp = np.abs(true_state - xksp)
ksplt=np.linspace(0,200,201)

#Plotting
fig3, ax3 = plt.subplots(2,2)
for i in (0,1):
    for j in (0,1):
        n = 2*1+j 
        ax3[i,j].scatter(ksplt,est_st_errp[n,:],s=2,label='Aircraft A')
        if j%2 == 0:
            ax3[i,j].scatter(ksplt,est_st_errp[4+n,:],s=2,label='Aircraft B')
        else:
            ax3[i,j].scatter(ksplt[1:],est_st_errp[4+n,1:],s=2,label='Aircraft B')
        ax3[i,j].legend()
        if n%2==0:
            ax3[i,j].set_ylabel('Position Error (m)')
            ax3[i,j].set_xlabel('Time (s)')
        else:
            ax3[i,j].set_ylabel('Velocity Error (m/s)')
            ax3[i,j].set_xlabel('Time (s)')
        ax3[0,j].set_title('East Error')
        ax3[1,j].set_title('North Error')
plt.suptitle('Aircraft Estimation Error')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Part C ii --------------------------------------------------------------------------------------
Hd = Hp = np.array([   [1, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, -1, 0]   ])
Fd = Fp
Qd = Qp

#initialize xhato and Po
xksd=[]
twosigmasd=[]
for i in range(201):
    if i==0:
        x_k_plus = xhat_init
        p_k_plus = p_init
   
    #time update step
    x_k_minus = Fd@x_k_plus
    p_k_min = Fd @ p_k_plus @ Fd.T + Qd
    k_k = p_k_min @ Hd.T @ LA.inv(Hd @ p_k_min @ Hd.T + R_d)
    
    #measurement update step
    x_k_plus = x_k_minus + (k_k @ (y_D[:,i] - Hp @ x_k_minus))
    p_k_plus = (np.identity(8) - k_k @ Hd) @ p_k_min
    twosigmasd.append([2*np.sqrt(p_k_plus[0,0]),2*np.sqrt(p_k_plus[1,1]),\
                      2*np.sqrt(p_k_plus[2,2]),2*np.sqrt(p_k_plus[3,3]),\
                      2*np.sqrt(p_k_plus[4,4]),2*np.sqrt(p_k_plus[5,5]),\
                      2*np.sqrt(p_k_plus[6,6]),2*np.sqrt(p_k_plus[7,7])])
    xksd.append(x_k_plus)
xksd=np.array(xksd).T
twosigmasd=np.array(twosigmasd).T
est_st_errd = np.abs(true_state - xksd)

#Plotting
fig4, ax4 = plt.subplots(2,2)
for i in (0,1):
    for j in (0,1):
        n = 2*1+j 
        ax4[i,j].scatter(ksplt,est_st_errd[n,:],s=2,label='Aircraft A')
        if j%2 == 0:
            ax4[i,j].scatter(ksplt,est_st_errd[4+n,:],s=2,label='Aircraft B')
        else:
            ax4[i,j].scatter(ksplt[1:],est_st_errd[4+n,1:],s=2,label='Aircraft B')
        ax4[i,j].legend()
        if n%2==0:
            ax4[i,j].set_ylabel('Position Error (m)')
            ax4[i,j].set_xlabel('Time (s)')
        else:
            ax4[i,j].set_ylabel('Velocity Error (m/s)')
            ax4[i,j].set_xlabel('Time (s)')
        ax4[0,j].set_title('East Error')
        ax4[1,j].set_title('North Error')
plt.suptitle('Aircraft Estimation Error - Only Transponder')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
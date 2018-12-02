import numpy as np
from scipy import linalg as LA
from numpy import linalg as LA2
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal

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
fig, ax = plt.subplots(1,3)
ax.plot(y[0,0:41],y[1,0:41],label="y")
ax.plot(xks[0,0:41],xks[2,0:41],label="x")
ax.plot()
ax.legend()
plt.show()

#calculate estimated state error
est_st_err = np.abs(x_val - xks)

#plot estimated state error
ksplt=np.linspace(0,200,201)
plt.subplot(221)
plt.scatter(ksplt,est_st_err[0,:],s=2)
plt.ylabel('East Error (m)')
plt.xlabel('Time (s)')
plt.subplot(222)
plt.scatter(ksplt,est_st_err[1,:],s=2)
plt.ylabel('East Vel. Error (m/s)')
plt.xlabel('Time (s)')
plt.subplot(223)
plt.scatter(ksplt,est_st_err[2,:],s=2)
plt.ylabel('North Error (m)')
plt.xlabel('Time (s)')
plt.subplot(224)
plt.scatter(ksplt,est_st_err[3,:],s=2)
plt.ylabel('North Vel. Error (m/s)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

#plot sigma bounds
plt.suptitle('2 * Sigma Bounds')
plt.subplot(221)
plt.scatter(ksplt,twosigmas[0,:],s=1)
plt.ylabel('East (m^2)')
plt.xlabel('Time (s)')
plt.subplot(222)
plt.scatter(ksplt,twosigmas[1,:],s=1)
plt.ylabel('East vel. (m/s)^2')
plt.xlabel('Time (s)')
plt.subplot(223)
plt.scatter(ksplt,twosigmas[2,:],s=1)
plt.ylabel('North (m^2)')
plt.xlabel('Time (s)')
plt.subplot(224)
plt.scatter(ksplt,twosigmas[3,:],s=1)
plt.ylabel('North vel. (m/s)^2')
plt.xlabel('Time (s)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#part c


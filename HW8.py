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

#Plotting
fig, ax = plt.subplots(1,1)
ax.plot(y[0,0:41],y[1,0:41])
plt.show()
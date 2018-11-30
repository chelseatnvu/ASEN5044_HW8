import numpy as np
from scipy import linalg as LA
from numpy import linalg as LA2
from matplotlib import pyplot as plt

O = 0.045
q_w = 10
w_a = q_w * np.array([[2,       0.05    ],
                      [0.05,    0.5     ]])
gamma_a = np.array([[0,0],
                    [1,0],
                    [0,0],
                    [0,1]])
A = np.array([[0,   1,  0,  0   ],
              [0,   0,  0,  -O  ],
              [0,   0,  0,  1   ],
              [0,   O,  0,  0   ]  ])
import numpy as np

def eom(x,t,mu):
    X,Xd,Y,Yd = x
    r = np.sqrt(X**2+Y**2)
    return [Xd,-mu*X/(r**3),Yd,-mu*Y/(r**3)]

def A(state,mu):
    x,xd,y,yd = state
    r = np.sqrt(x**2+y**2)
    return np.array(
            [[0,                            1,      0,                          0   ],
             [3*mu*x**2/r**5 - mu/r**3,     0,      3*mu*x*y/r**5,              0   ],
             [0,                            0,      0,                          1   ],
             [3*mu*x*y/r**5,                0,      3*mu*y**2/r**5 - mu/r**3,   0   ]])
    
def station(i,t):
    R_e = 6378
    w_e = 2*np.pi/86400
    th = (i-1)*np.pi/6
    x = R_e*np.cos(w_e*t+th)
    y = R_e*np.sin(w_e*t+th)
    xd = -R_e*w_e*np.sin(w_e*t+th)
    yd = R_e*w_e*np.cos(w_e*t+th)
    return np.array([x,xd,y,yd]).reshape(4)
    
def Ci(state,station_state,mu):
    x,xd,y,yd = state
    x_station,xd_station,y_station,yd_station = station_state
    rho = np.sqrt((x-x_station)**2+(y-y_station)**2)
    rx = x-x_station
    ry = y-y_station
    rxd = xd-xd_station
    ryd = yd-yd_station
    return np.array(
            [[rx/rho,                               0,      ry/rho,                                 0       ],
             [rxd/rho - rx*(rxd*rx+ry*ryd)/rho**3,  rx/rho, ryd/rho - ry*(ryd*ry+rxd*rx)/rho**3,    ry/rho  ],
             [-ry/rho**2,                           0,      rx/rho**2,                              0       ]])
    
def C(state,mu,t,pings):
    C = []
    for i in pings:
        C.append(Ci(state,station(i,t),mu))
    return np.array(C).reshape(int(np.size(C)/4),4)

def y(state,mu,t,pings):
    x_c,xd_c,y_c,yd_c = state
    y = []
    for i in pings:
        x_st,xd_st,y_st,yd_st = station(i,t)
        th = np.arctan2(y_st,x_st)
        rho = np.sqrt((x_c-x_st)**2+(y_c-y_st)**2)
        rho_d = ((x_c-x_st)*(xd_c-xd_st) + (y_c-y_st)*(yd_c-yd_st))/rho
        phi = np.arctan2((y_c-y_st),(x_c-x_st))
        y.append(np.array([[rho,rho_d,phi,i]]).T)
    return np.array(y).reshape(int(np.size(y)),1)

#state=np.array([-2164.31,-7.2572,6328.43,7.5])
#mu = 3.986004415e5
#t = 1660
#testC = C(state,mu,t) @ state.T
#testy = y(state,mu,t)
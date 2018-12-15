import numpy as np
import scipy as sp

def eom(x,t,mu,noise,step):
    X,Xd,Y,Yd = x
    r = np.sqrt(X**2+Y**2)
    dirt = noise[int(t/step),:]
    return [Xd,-mu*X/(r**3)+dirt[0],Yd,-mu*Y/(r**3)+dirt[1]]

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
    xs,xds,ys,yds = station_state
    rho = np.sqrt((x-xs)**2+(y-ys)**2)
    rx = x-xs
    ry = y-ys
    rxd = xd-xds
    ryd = yd-yds
    return np.array(
            [[rx/rho,                               0,      ry/rho,                                 0       ],
             [(ry*(-rx*ryd+xd*ry-xds*ry))/rho**3,   rx/rho, (rx*(rx*ryd+xds*ry-xd*ry))/rho**3,      ry/rho  ],
             [-ry/rho**2,                           0,      rx/rho**2,                              0       ]])
    
def C(state,mu,t,pings):
    C = []
    for i in pings:
        C.append(Ci(state,station(i,t),mu))
    return np.array(C).reshape(int(np.size(C)/4),4)

def H_mat(state,mu,t,pings):
    y_val = y(state,mu,t,pings)
    if len(y_val) == 5:
        epsil = [1e-6,1e-12,1e-6,1e-12]
        ans = np.empty((3,4))
        for i in np.arange(0,3):
            for j in 0,1,2,3:
                addit = (epsil[i]*np.eye(4)[:,j]).reshape(4,1)
                state2 = (state+addit).reshape(4,1)
                ans[i,j] = ( y(state2,mu,t,pings)[i,0] - y(state,mu,t,pings)[i,0] ) / epsil[i]
        return ans
    elif len(y_val) == 10:
        epsil = [1e-6,1e-12,1e-6,1e-12,1e-6,1e-12,1e-6,1e-12]
        ans = np.empty((6,4))
        k = 0
        for i in 0,1,2,5,6,7:
            for j in 0,1,2,3:
                addit = (epsil[k]*np.eye(4)[:,j]).reshape(4,1)
                state2 = (state+addit).reshape(4,1)
                ans[k,j] = ( y(state2,mu,t,pings)[k,0] - y(state,mu,t,pings)[k,0] ) / epsil[k]
            k += 1
        return ans
    

def y(state,mu,t,pings):
    x_c,xd_c,y_c,yd_c = state
    y = []
    for i in pings:
        x_st,xd_st,y_st,yd_st = station(i,t)
        th = np.arctan2(y_st,x_st)
        rho = np.sqrt((x_c-x_st)**2+(y_c-y_st)**2)
        rho_d = ((x_c-x_st)*(xd_c-xd_st) + (y_c-y_st)*(yd_c-yd_st))/rho
        phi = np.arctan2((y_c-y_st),(x_c-x_st))
        y.append(np.array([[rho,rho_d,phi,i,t]]).T)
    return np.array(y).reshape(int(np.size(y)),1)

def measure(state,mu,t):
    x_c,xd_c,y_c,yd_c = state
    y = []
    for i in np.arange(1,13):
        x_st,xd_st,y_st,yd_st = station(i,t)
        th = np.arctan2(y_st,x_st)
        rho = np.sqrt((x_c-x_st)**2+(y_c-y_st)**2)
        rho_d = ((x_c-x_st)*(xd_c-xd_st) + (y_c-y_st)*(yd_c-yd_st))/rho
        phi = np.arctan2((y_c-y_st),(x_c-x_st))
        if -np.pi/2 < phi-th < np.pi/2:
            y.append(np.array([[rho,rho_d,phi,i]]).T)
    return np.array(y).reshape(int(np.size(y)),1)

def gen_meas(x0,end,step,mu,dirt,R):
    x_clean = sp.integrate.odeint(eom,x0,np.arange(0,end+step,step),args=(mu,dirt,step))
    cov = R
    noise = np.random.multivariate_normal([0,0,0],cov,size=20000).reshape(20000,3).T
    noise = np.append(noise,np.zeros((1,20000)),axis=0)
    y_clean = []
    y_dirty = []
    for t in np.arange(0,end+step,step):
        k = int(t/step)
        state = x_clean[k,:]
        clean = measure(state,mu,t)
        clean = clean.reshape(len(clean),1)
        y_clean.append(clean)
        if len(clean) == 4:
            y_dirty.append(clean+noise[:,k].reshape(4,1))
        elif len(clean) == 8:
            second = np.random.multivariate_normal([0,0,0],cov,size=(1)).reshape(1,3).T
            second = np.append(second,[[0]],axis=0).reshape(4,1)
            y_dirty.append(clean+np.append(noise[:,k].reshape(4,1),second,axis=0))
        elif len(clean) == 0:
            y_dirty.append(np.array([]))
        else: print('Hold Up')
    return y_dirty

#state=np.array([7000,0,0,7.5])
#mu = 3.986004415e5
##t = 1660
##testC = C(state,mu,t) @ state.T
#x0 = np.array([6678,0,0,6678*np.sqrt(mu/6678**3)])
#testy = gen_meas(x0,10000,10,mu)
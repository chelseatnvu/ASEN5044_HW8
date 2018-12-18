# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:10:46 2018

@author: Chelsea
"""
import numpy as np
from scipy.integrate import odeint

mu = 398600 #km^3/s^2
Re = 6.378e3  #km
we = 2*np.pi/86400  #rad/s
dt = 10 #s

#Solver functions
def eom(rv,t):
    x,xvel,y,yvel=rv
    r=np.sqrt(x**2+y**2)
    xdot=xvel
    vxdot=-mu*x/(r**3)
    ydot=yvel
    vydot=-mu*y/(r**3)
    drvdt=[xdot,vxdot,ydot,vydot]
    return drvdt

def eom2(rv,t,noise,step):
    x,xvel,y,yvel=rv
    r=np.sqrt(x**2+y**2)
    xdot=xvel
    vxdot=-mu*x/(r**3)
    ydot=yvel
    vydot=-mu*y/(r**3)
    ind = int(t/dt)
    if ind>=noise[:,0].size:
        ind = noise[:,0].size-1
    drvdt=[xdot,vxdot+noise[ind,0],ydot,vydot+noise[ind,1]]
    return drvdt

def solvetraj(x,y,xvel,yvel,TOF):
    rv0=[x,xvel,y,yvel]    #initial conditions for odeint
    t=np.arange(0.,TOF,dt)    #(start, stop, stepsize)
    sol= odeint(eom,rv0,t)#,full_output=1)
    return [sol,t]

def solvetraj2(x,y,xvel,yvel,TOF,noise):
    rv0=[x,xvel,y,yvel]    #initial conditions for odeint
    t=np.arange(0.,TOF,dt)    #(start, stop, stepsize)
    sol= odeint(eom2,rv0,t,args=(noise,dt))#,full_output=1)
    return [sol,t]


def genA(X,Y):
    r = np.sqrt(X**2+Y**2)
    A = np.array([[0.,                            1., 0.,                         0.],
                  [3*mu*X**2/r**5 - mu/r**3,      0., 3*mu*X*Y/r**5,              0.],
                  [0.,                            0., 0.,                         1.],
                  [3*mu*X*Y/r**5,                 0., 3*mu*Y**2/r**5 - mu/r**3,   0.]])
    return A

def genB():
    B = np.array([[0, 0],\
              [1,0],\
              [0, 0],\
              [0,1]])
    return B
    
#generate C matrix for certain station with coordinates [Xs,Ys] and vel [Xsd,Ysd]
def genCi(X,Y,Xd,Yd,Xs,Ys,Xsd,Ysd):
    rho = np.sqrt((X-Xs)**2+(Y-Ys)**2)
    C = np.array([[(X-Xs)/rho, 0, (Y-Ys)/rho, 0],\
                       [(Xd-Xsd)/rho - (X-Xs)*((Xd-Xsd)*(X-Xs)+(Y-Ys)*(Yd-Ysd))/rho**3,\
                   (X-Xs)/rho,\
                   (Yd-Ysd)/rho - (Y-Ys)*((Yd-Ysd)*(Y-Ys)+(Xd-Xsd)*(X-Xs))/rho**3,\
                   (Y-Ys)/rho],\
                    [-(Y-Ys)/rho**2, 0, (X-Xs)/rho**2, 0]])
    return C


#define function that returns position and velocity of ith station
#inputs: i -> station index [1,12], t-> time to evaluate state at
def statestation(i,t):
    theta0 = (i-1)*np.pi/6
    x = Re * np.cos(we*t+theta0)
    y = Re * np.sin(we*t+theta0)
    xsd = -Re * we * np.sin(we*t+theta0)
    ysd = Re * we* np.cos(we*t+theta0)
    return x,xsd,y,ysd

def genCstack2(x_nom,IDs,t):
    X0,Xd0,Y0,Yd0 = x_nom
    C=[]
    y_star=[]
    for i in IDs:
        Xs,Xsd,Ys,Ysd = statestation(i,t)
        rho = np.sqrt((X0-Xs)**2+(Y0-Ys)**2)
        rho_rate = ((X0 - Xs) * (Xd0 - Xsd) + (Y0 - Ys) * (Yd0 - Ysd)) / rho
        el = np.arctan2((Y0-Ys),(X0-Xs))
        Ci = genCi(X0,Y0,Xd0,Yd0,Xs,Ys,Xsd,Ysd)
        C.append(Ci)
        y_star.append([rho,rho_rate,el,i])
    if not C:   #if list is empty return empty array
        return np.array([]), np.array([])
    else:
        C = np.concatenate(C, axis = 0)
        return C
    

def genCstack(X0,Y0,Xd0,Yd0,t):
    C = []
    y_star = []
    for i in range(12):
        Xs,Ys,Xsd,Ysd = statestation(i+1,t)
        #only generate C if elevation is 0 or higher
        #calculate elevation
        el = np.arctan2((Y0-Ys),(X0-Xs))
        thet = np.arctan2(Ys,Xs)
        if el>=(-np.pi/2+thet) and el<(np.pi/2+thet):
            Ci, rho, rho_rate = genCi(X0,Y0,Xd0,Yd0,Xs,Ys,Xsd,Ysd)
            C.append(Ci)
            y_star.append([rho,rho_rate,el,i+1])
    y_star = np.array(y_star).T
    if not C:   #if list is empty return empty array
        return np.array([]), np.array([])
    else:
        C = np.concatenate(C, axis = 0)
        return C

def gengamma():
    gamma = np.array([[0, 0],\
                      [1, 0],\
                      [0, 0],\
                      [0, 1]])
    return gamma

def meas(state_sc,IDs,time):
    y_n = []
    for i in IDs:
        xs,xsd,ys,ysd = statestation(i,time)
        x,xd,y,yd = state_sc
        rho = np.sqrt((x-xs)**2+(y-ys)**2)
        rho_d = ((x-xs)*(xd-xsd)+(y-ys)*(yd-ysd))/rho
        phi = np.arctan2((y-ys),(x-xs))
        y_n.append([rho,rho_d,phi,i])
#    y_n = np.array(y_n)
    return y_n

def noisymeas(state_sc,IDs,time,noise):
    y_n = []
    for i in IDs:
        xs,xsd,ys,ysd = statestation(i,time)
        x,xd,y,yd = state_sc
        rho = np.sqrt((x-xs)**2+(y-ys)**2)
        rho_d = ((x-xs)*(xd-xsd)+(y-ys)*(yd-ysd))/rho
        phi = np.arctan2((y-ys),(x-xs))
        y_n.append([rho+noise[0],rho_d+noise[1],phi+noise[2]])
#    y_n = np.array(y_n)
    return y_n

#build nominal measurements. add measurement noise to x_star
def geny_nom(R,TOF,time,x_star):
    meas_noise = np.random.multivariate_normal([0,0,0],R,size=int(TOF/dt))
    y_nom = []
    y_nom_ns = []
    vis = []
#    for i in range(np.shape(x_star)[0]):
    for i in range(len(time)):
        #find which stations are visible
        IDs_vis = []
        for j in range(12):
            xs,xsd,ys,ysd = statestation(j+1,time[i])        
            thet = np.arctan2(ys,xs)
            phi = np.arctan2((x_star[i,2] - ys),(x_star[i,0] - xs))
            ang_diff = phi - thet
            if ang_diff > np.pi:
                ang_diff = 2*np.pi - ang_diff
            elif ang_diff < -np.pi:
                ang_diff = 2*np.pi + ang_diff
            if -np.pi/2 <= ang_diff <= np.pi/2:
                IDs_vis.append(j+1)
        if len(IDs_vis)<1:
            y_nom_ns.append([])
            y_nom.append([])
            vis.append([])
        else:
            y_nom.append(np.array(meas(x_star[i,:],IDs_vis,time[i]))[:,0:3])
            y_nom_ns.append(np.array(noisymeas(x_star[i,:],IDs_vis,time[i],meas_noise[i,:])))
            vis.append(IDs_vis)
    return y_nom_ns, y_nom, meas_noise, vis

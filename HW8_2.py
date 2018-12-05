# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:52:39 2018

@author: Chelsea
"""
import numpy as np
from scipy import linalg as LA

#part b
#nominal conditions
#state of s/c is defined as [X,Xd,Y,Yd]
#station located at [Xs,Ys] with velocity [Xsd,Ysd]
#constants
mu = 398600 #km^3/s^2
Re = 6.378e3  #km
we = 2*np.pi/86400  #rad/s
#initial position of s/c
X0 = 6678   #km
Y0 = 0  #km
ro = np.sqrt(X0**2+Y0**2)
Xd0 = 0     #km/s
Yd0 = ro * np.sqrt(mu/ro**3)
#define function that returns position and velocity of ith station
#inputs: i -> station index [1,12], t-> time to evaluate state at
def statestation(i,t):
    theta0 = (i-1)*np.pi/6
    x = Re * np.cos(we*t+theta0)
    y = Re * np.sin(we*t+theta0)
    return x,y

#CT state matrices
def genA(X,Y):
    A = np.array([[0., 1., 0., 0.],\
         [(-mu*(X+Y)**1.5 + 1.5*mu*X*(X+Y)**.5)/((X+Y)**3), 0., 1.5*mu*X*(X+Y)**-2.5, 0.],\
         [0., 0., 0., 1.],\
         [1.5*mu*Y*(X+Y)**(-2.5), 0., (-mu*(X+Y)**1.5+ 1.5* mu*Y*(X+Y)**.5)/((X+Y)**3),0.]])
    return A
A = genA(X0,Y0)
B = np.array([[0, 0],\
              [1,0],\
              [0, 0],\
              [0,1]])
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
C = []
for i in range(12):
    t = 0   #set t = 0 because nominal conditions are at t = 0.
    Xs,Ys = statestation(i+1,t)
    #only generate C if elevation is 0 or higher
    #calculate elevation
    el = np.arctan2((Y0-Ys),(X0-Xs))
    if el >=0:
        thet = np.arctan2(Ys,Xs)
        Xsd = we * np.cos(thet)
        Ysd = we * np.sin(thet)
        C.append(genCi(X0,Y0,Xd0,Yd0,Xs,Ys,Xsd,Ysd))
C = np.concatenate(C, axis = 0)
D = np.zeros((np.shape(C)[0],2))
gamma = np.array([[0, 0],\
                  [1, 0],\
                  [0, 0],\
                  [0, 1]])
w = np.array([[1.,1.],[1.,1.]]) #just a guess?

dt = 10. #s
Z = np.append(-A,gamma@w@gamma.T,axis=1)
Z = dt*np.append(Z,np.append(np.zeros((4,4)),A.T,axis=1),axis=0)
expZ = LA.expm(Z)

F = expZ[4:,4:].T
Q = F @ expZ[0:4,4:]

#find G
A_hat = np.append(A,B,axis=1)
A_hat = np.append(A_hat,np.zeros((2,6)),axis=0)
exp_A_hat = LA.expm(A_hat)
G = exp_A_hat[0:4,4:]

#define H and M. they are the same as C and D evaluated at nominal respectively
H = C
M = D

#propagate forward assuming no inputs (so assume G = 0)
xs_DT = [[X0,Y0,Xd0,Yd0]]
for i in range(5431):
    xs_DT.append(F @ xs_DT[i])
xs_DT[0]=np.array([X0,Y0,Xd0,Yd0])
xs_DT=np.array(xs_DT)
    
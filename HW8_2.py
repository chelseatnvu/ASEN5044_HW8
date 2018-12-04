# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:52:39 2018

@author: Chelsea
"""
import numpy as np
#part b
#nominal conditions
#constants
mu = 398600 #km^3/s^2
Re = 6.378e3  #km
we = 2*np.pi/86400  #rad/s

#define function that returns position and velocity of ith station
#inputs: i -> station index [1,12], t-> time to evaluate state at
def statestation(i,t):
    theta0 = (i-1)*np.pi/6
    x = Re * np.cos(we*t+theta0)
    y = Re * np.sin(we*t+theta0)
    return x,y
#state is defined as [X,Xd,Y,Yd]
#station located at [Xs,Ys] with velocity [Xsd,Ysd]
#CT state matrices
A = np.array([[0, 1, 0, 0],\
              [(-mu(X+Y)**1.5 - 1.5*mu*X*(X+Y)**.5)/((X+Y)**3), 0, 1.5*mu*X*(X+Y)**-2.5, 0],\
                [0, 0, 0, 1],\
                [1.5*mu*Y*(X+Y)**(-2.5), 0, (-mu*(X+Y)**1.5-mu*Y*(X+Y)**.5)/((X+Y)**3)]])
B = np.array([[0, 0],\
              [1,0],\
              [0, 0],\
              [0,1]])
rho = np.sqrt((X-Xs)**2+(Y-Ys)**2)
#generate C matrix for certain station with coordinates [Xs,Ys] and vel [Xsd,Ysd]
def genCi(Xs,Ys,Xsd,Ysd):
    #velocity of station on equator = 2*pi*r_e/1 day
    C = np.array([[(X-Xs)/rho, 0, (Y-Ys)/rho, 0],\
                   [(Xd-Xsd)/rho - (X-Xs)*((Xd-Xsd)*(X-Xs)+(Y-Ys)*(Yd-Ysd))/rho**3,\
                   (X-Xs)/rho,\
                   (Yd-Ysd)/rho - (Y-Ys)*((Yd-Ysd)*(Y-Ys)+(Xd-Xsd)*(X-Xs))/rho**3,\
                   (Y-Ys)/rho],\
                    [-(Y-Ys)/rho**2, 0, (X-Xs)/rho**2, 0]])
    return C
C = np.empty((3*12,4))
for i in range(12):
    Xs,Ys = statestation(i+1,t)
    thet = np.arctan2(Ys,Xs)
    Xsd = we * np.cos(thet)
    Ysd = we * np.sin(thet)
    C[3*i:3*(i+1)] = genCi(Xs,Ys,Xsd,Ysd)
D = np.zeros((3,4))
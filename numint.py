# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:46:05 2018

@author: Chelsea
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
mu = 398600 #km^3/s^2

def eom(rv,t):
	x,y,xvel,yvel=rv
	r=np.sqrt(x**2+y**2)
	xdot=xvel
	vxdot=-mu*x/(r**3)
	ydot=yvel
	vydot=-mu*y/(r**3)
	drvdt=[xdot,ydot,vxdot,vydot]
	return drvdt

def solvetraj(x,y,xvel,yvel,TOF):
	rv0=[x,y,xvel,yvel]	#initial conditions for odeint
	t=np.arange(0.,TOF,10.)	#(start, stop, stepsize)
	sol=odeint(eom,rv0,t)#,full_output=1)
	return [sol,t]

    
ro=np.array([6678.,0.])    #km
ro_mag = np.linalg.norm(ro)
vo=np.array([0.,ro_mag*np.sqrt(mu/ro_mag**3)])    #km/s
print('initial conditions:', ro,' km',vo,' km/s')

#calculate period of orbit
n = np.sqrt(mu/ro_mag**3)
T = 2*np.pi/n
print('period = ',T,' s')

a=solvetraj(ro[0],ro[1],vo[0],vo[1],T+10.)

#plot orbit
fig = plt.figure()
ax = fig.gca(projection='3d')
z = a[0][:,2]
x = a[0][:,0]
y = a[0][:,1]
ax.plot(x, y, z, label='orbit')
plt.xlabel('x (km)')
plt.ylabel('y (km)')
# plt.zlabel('z')
ax.legend()

plt.show()
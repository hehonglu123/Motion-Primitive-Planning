
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
import sys
import numpy as np



data = read_csv("greedy_fitting/comparison/moveL+moveC/threshold1/curve_fit_backproj.csv")
curve_x=data['x'].tolist()
curve_y=data['y'].tolist()
curve_z=data['z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

print(curve.shape)

###plot results
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray')
breakpoints=[0,8323,15198,20027,25099,31777,38996,45083,47936,49999,-1]

ax.plot3D(curve[breakpoints[0]:breakpoints[1],0], curve[breakpoints[0]:breakpoints[1],1], curve[breakpoints[0]:breakpoints[1],2], c='blue')
ax.plot3D(curve[breakpoints[1]:breakpoints[2],0], curve[breakpoints[1]:breakpoints[2],1], curve[breakpoints[1]:breakpoints[2],2], c='blue')
ax.plot3D(curve[breakpoints[2]:breakpoints[3],0], curve[breakpoints[2]:breakpoints[3],1], curve[breakpoints[2]:breakpoints[3],2], c='blue')
ax.plot3D(curve[breakpoints[3]:breakpoints[4],0], curve[breakpoints[3]:breakpoints[4],1], curve[breakpoints[3]:breakpoints[4],2], c='red')
ax.plot3D(curve[breakpoints[4]:breakpoints[5],0], curve[breakpoints[4]:breakpoints[5],1], curve[breakpoints[4]:breakpoints[5],2], c='blue')
ax.plot3D(curve[breakpoints[5]:breakpoints[6],0], curve[breakpoints[5]:breakpoints[6],1], curve[breakpoints[5]:breakpoints[6],2], c='blue')
ax.plot3D(curve[breakpoints[6]:breakpoints[7],0], curve[breakpoints[6]:breakpoints[7],1], curve[breakpoints[6]:breakpoints[7],2], c='blue')
ax.plot3D(curve[breakpoints[7]:breakpoints[8],0], curve[breakpoints[7]:breakpoints[8],1], curve[breakpoints[7]:breakpoints[8],2], c='red')
ax.plot3D(curve[breakpoints[8]:breakpoints[9],0], curve[breakpoints[8]:breakpoints[9],1], curve[breakpoints[8]:breakpoints[9],2], c='red')
ax.plot3D(curve[breakpoints[-1]:-1,0], curve[breakpoints[-1]:-1,1], curve[breakpoints[-1]:-1,2], c='brown')

ax.scatter(curve[breakpoints,0], curve[breakpoints,1], curve[breakpoints,2])

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')



# ax.scatter3D(curve[4200:4400,0], curve[4200:4400,1], curve[4200:4400,2], c=curve[4200:4400,2], cmap='Blues')

plt.show()
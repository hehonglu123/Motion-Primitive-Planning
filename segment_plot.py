
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
import sys
import numpy as np



col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("data/from_interp/Curve_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T



###plot results
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1], curve[:,2], 'gray')
breakpoints=[0, 427, 853, 1278, 1714, 2166, 2643, 3154, 3713, 4344, 5095, 5985, 7404, 8367, 9128, 9780, 10360, 10890, 11383, 11850, 12300, 12740, 13180, 13581, 13990, 14415, 14860, 15332, 15835, 16376, 16944, 17580]

ax.plot3D(curve[breakpoints[2]:breakpoints[3],0], curve[breakpoints[2]:breakpoints[3],1], curve[breakpoints[2]:breakpoints[3],2], 'red')
ax.plot3D(curve[breakpoints[5]:breakpoints[6],0], curve[breakpoints[5]:breakpoints[6],1], curve[breakpoints[5]:breakpoints[6],2], 'red')
ax.plot3D(curve[breakpoints[8]:breakpoints[9],0], curve[breakpoints[8]:breakpoints[9],1], curve[breakpoints[8]:breakpoints[9],2], 'red')
ax.plot3D(curve[breakpoints[17]:breakpoints[18],0], curve[breakpoints[17]:breakpoints[18],1], curve[breakpoints[17]:breakpoints[18],2], 'red')
ax.plot3D(curve[breakpoints[18]:breakpoints[19],0], curve[breakpoints[18]:breakpoints[19],1], curve[breakpoints[18]:breakpoints[19],2], 'red')
ax.plot3D(curve[breakpoints[19]:breakpoints[20],0], curve[breakpoints[19]:breakpoints[20],1], curve[breakpoints[19]:breakpoints[20],2], 'red')
ax.plot3D(curve[breakpoints[20]:breakpoints[21],0], curve[breakpoints[20]:breakpoints[21],1], curve[breakpoints[20]:breakpoints[21],2], 'red')
ax.plot3D(curve[breakpoints[22]:breakpoints[23],0], curve[breakpoints[22]:breakpoints[23],1], curve[breakpoints[22]:breakpoints[23],2], 'red')
ax.plot3D(curve[breakpoints[23]:breakpoints[24],0], curve[breakpoints[23]:breakpoints[24],1], curve[breakpoints[23]:breakpoints[24],2], 'red')


# ax.scatter3D(curve[4200:4400,0], curve[4200:4400,1], curve[4200:4400,2], c=curve[4200:4400,2], cmap='Blues')

plt.show()
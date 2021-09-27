import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from toolbox_circular_fit import *

###read in points
col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../data/from_interp/Curve_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

break_point=int(len(curve)/2)
curve1=curve[:break_point]
curve2=curve[break_point:]

curve_fitarc1,curve_fitcircle1=circle_fit(curve1,p=curve[break_point])
curve_fitarc2,curve_fitcircle2=circle_fit(curve2,p=curve[break_point])

###error calc
fit=np.vstack((curve_fitarc1,curve_fitarc2))
error=[]
for i in range(len(fit)):
    error_temp=np.linalg.norm(curve-fit[i],axis=1)
    idx=np.argmin(error_temp)
    error.append(error_temp[idx])

error=np.array(error)
max_cartesian_error=np.max(error)
avg_cartesian_error=np.average(error)

print('max error: ',max_cartesian_error)
print('average error: ',avg_cartesian_error)

###3D plot
plt.figure()
ax = plt.axes(projection='3d')
# ax.set_xlim3d(1500, 3000)
# ax.set_ylim3d(-500, 1000)
# ax.set_zlim3d(0, 1500)


ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')

# ax.scatter3D(curve[:,0], curve[:,1], curve[:,2], c=curve[:,2], cmap='Reds')

# ax.scatter3D(curve_fitcircle1[:,0], curve_fitcircle1[:,1], curve_fitcircle1[:,2], c=curve_fitcircle1[:,2], cmap='Greens')
# ax.scatter3D(curve_fitcircle2[:,0], curve_fitcircle2[:,1], curve_fitcircle2[:,2], c=curve_fitcircle2[:,2], cmap='Blues')

ax.scatter3D(curve_fitarc1[:,0], curve_fitarc1[:,1], curve_fitarc1[:,2], c=curve_fitarc1[:,2], cmap='Greens')
ax.scatter3D(curve_fitarc2[:,0], curve_fitarc2[:,1], curve_fitarc2[:,2], c=curve_fitarc2[:,2], cmap='Blues')

ax.scatter3D(curve[break_point][0], curve[break_point][1], curve[break_point][2], c=curve[break_point][2], cmap='Oranges_r')



plt.show()

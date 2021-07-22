
import numpy as np
from pandas import *
import sys
from general_robotics_toolbox import *


col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("Curve.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()


###reference frame transformation
R=np.array([[0,0,1.],
			[1.,0,0],
			[0,1.,0]])
T=np.array([[2700.],[-800.],[500.]])
H=np.vstack((np.hstack((R,T)),np.array([0,0,0,1])))
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_base=np.zeros(curve.shape)
###checkpoint1
# print(np.dot(R,direction2R(curve_direction[0],curve[1]-curve[0])))
for i in range(len(curve)):
	curve_base[i]=np.dot(H,np.hstack((curve[i],[1])).T)[:-1]

df=DataFrame({'x':curve_base[:,0],'y':curve_base[:,1], 'z':curve_base[:,2]})
df.to_csv('Curve_in_base_frame.csv',header=False,index=False)
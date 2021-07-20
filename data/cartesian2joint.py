import numpy as np
from pandas import *
import sys
from general_robotics_toolbox import *

sys.path.append('../toolbox')
from robot_def import *

def cross(v):
	return np.array([[0,-v[-1],v[1]],
					[v[-1],0,-v[0]],
					[-v[1],v[0],0]])
def direction2R(v):
	theta = np.arccos(np.dot(v,np.array([0,0,1])))
	print(theta)
	return rot(np.cross(v,np.array([0,0,1])),theta)


col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("Curve_interp.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()

curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

curve_base=np.zeros(curve.shape)
curve_R_base=[]
# curve_base[:,0]=2700+curve[:,2]
# curve_base[:,1]=-800+curve[:,0]
# curve_base[:,2]=500+curve[:,1]


R=np.array([[0,0,1],
			[1,0,0],
			[0,1,0]])
T=np.array([[2700],[-800],[500]])
H=np.vstack((np.hstack((R,T)),np.array([0,0,0,1])))

print(direction2R(curve_direction[0]))
for i in range(len(curve)):
	curve_base[i]=np.dot(H,np.hstack((curve[i],[1])).T)[:-1]
	# print(direction2R(curve_direction[i]))
	# curve_R_base.append()



# curve_base=curve_base/1000.

# curve_js=np.zeros((len(curve),6))

# for i in range(len(curve_base)):
# 	curve_js[i]=inv(curve_base[i])


# ###output to csv
# df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
# df.to_csv('Curve_js.csv')
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
def direction2R(v_norm,v_tang):
	v_norm=v_norm/np.linalg.norm(v_norm)
	theta1 = np.arccos(np.dot(v_norm,np.array([0,0,1])))
	###rotation to align z axis with opposite of curve normal
	axis_temp=np.cross(v_norm,np.array([0,0,1]))
	R1=rot(axis_temp/np.linalg.norm(axis_temp),theta1)

	###find correct x direction
	v_temp=v_tang-np.dot(v_tang,np.array([0,0,1]))

	theta2 = np.arccos(np.dot(v_temp/np.linalg.norm(v_temp),np.array([1,0,0])))
	###rotation about z axis to minimize x direction error
	R2=rot(np.sign(np.cross(v_temp/np.linalg.norm(v_temp),np.array([1,0,0]))),theta2)

	return np.dot(R2,R1)



col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("Curve_dense.csv", names=col_names)
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


###reference frame transformation
R=np.array([[0,0,1.],
			[1.,0,0],
			[0,1.,0]])
T=np.array([[2700.],[-800.],[500.]])
H=np.vstack((np.hstack((R,T)),np.array([0,0,0,1])))

###checkpoint1
# print(np.dot(R,direction2R(curve_direction[0],curve[1]-curve[0])))
for i in range(len(curve)):
	curve_base[i]=np.dot(H,np.hstack((curve[i],[1])).T)[:-1]
	try:
		R_curve=direction2R(curve_direction[i],curve[i+1]-curve[i])

	except:
		pass
	curve_R_base.append(np.dot(R,R_curve))
	


###units
curve_base=curve_base/1000.

curve_js=np.zeros((len(curve),6))
q_init=np.radians([35.406892, 12.788519, 27.907507, -89.251430, 52.417435, -128.363215])

for i in range(len(curve_base)):
	q_all=inv(curve_base[i],curve_R_base[i])

	###choose inv_kin closest to previous joints
	if i==0:
		temp_q=q_all-q_init
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		curve_js[i]=q_all[order[0]]
	else:
		temp_q=q_all-curve_js[i-1]
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		curve_js[i]=q_all[order[0]]
###checkpoint2
# print(np.degrees(curve_js[50]),curve_R_base[i])


###checkpoint3
# H=np.vstack((np.hstack((R.T,-np.dot(R.T,T))),np.array([0,0,0,1])))
# curve_base_temp=np.zeros(curve.shape)
# for i in range(len(curve_js)):
# 	curve_base_temp[i]=(np.dot(H,np.hstack((1000.*fwd(curve_js[i]).p,[1])).T)[:-1])
# print(curve_base_temp)




###output to csv
df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
df.to_csv('Curve_js.csv',header=False,index=False)
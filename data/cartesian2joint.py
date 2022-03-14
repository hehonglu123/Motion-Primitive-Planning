import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robots_def import *

def cross(v):
	return np.array([[0,-v[-1],v[1]],
					[v[-1],0,-v[0]],
					[-v[1],v[0],0]])
def direction2R(v_norm,v_tang):
	v_norm=v_norm/np.linalg.norm(v_norm)
	theta1 = np.arccos(np.dot(np.array([0,0,1]),v_norm))
	###rotation to align z axis with curve normal
	axis_temp=np.cross(np.array([0,0,1]),v_norm)
	R1=rot(axis_temp/np.linalg.norm(axis_temp),theta1)

	###find correct x direction
	v_temp=v_tang-v_norm * np.dot(v_tang, v_norm) / np.linalg.norm(v_norm)

	###get as ngle to rotate
	theta2 = np.arccos(np.dot(R1[:,0],v_temp/np.linalg.norm(v_temp)))


	axis_temp=np.cross(R1[:,0],v_temp)
	axis_temp=axis_temp/np.linalg.norm(axis_temp)

	###rotation about z axis to minimize x direction error
	R2=rot(np.array([0,0,np.sign(np.dot(axis_temp,v_norm))]),theta2)


	return np.dot(R1,R2)


def main():

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("from_ge/Curve_in_base_frame2.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


	abb6640_obj=abb6640(d=50)

	###back projection
	d=0			###offset
	# plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	# plt.show()

	# curve=curve-d*curve_direction

	curve_R=[]


	for i in range(len(curve)):
		try:
			R_curve=direction2R(curve_direction[i],-curve[i+1]+curve[i])
			if i>0:
				k,angle_of_change=R2rot(np.dot(curve_R[-1],R_curve.T))
				if angle_of_change>0.1:
					curve_R.append(curve_R[-1])
					continue
		except:
			traceback.print_exc()
			pass
		
		curve_R.append(R_curve)

	###insert initial orientation
	curve_R.insert(0,curve_R[0])
	curve_js=np.zeros((len(curve),6))

	# q_init=np.radians([35.414132, 12.483655, 27.914093, -89.255298, 51.405928, -128.026891])
	q_init=np.array([0.625835928,	0.836930134,	-0.239948016,	1.697010866,	-0.89108048,	0.800838687])
	for i in range(len(curve)):
		try:
			q_all=np.array(abb6640_obj.inv(curve[i],curve_R[i]))
		except:
			traceback.print_exc()
			pass
		###choose inv_kin closest to previous joints
		if i==0:
			temp_q=q_all-q_init
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			curve_js[i]=q_all[order[0]]

		else:
			try:
				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				curve_js[i]=q_all[order[0]]

			except:
				q_all=np.array(abb6640_obj.inv(curve[i],curve_R[i]))
				traceback.print_exc()
				break


	###checkpoint3
	###make sure fwd(joint) and original curve match
	# H=np.vstack((np.hstack((R.T,-np.dot(R.T,T))),np.array([0,0,0,1])))
	# curve_temp=np.zeros(curve.shape)
	# for i in range(len(curve_js)):
	# 	curve_temp[i]=(np.dot(H,np.hstack((fwd(curve_js[i]).p,[1])).T)[:-1])
	# print(np.max(np.linalg.norm(curve-curve_temp,axis=1)))




	###output to csv
	df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
	df.to_csv('from_ge/Curve_js2.csv',header=False,index=False)


def main2():
	sys.path.append('../constraint_solver')
	from constraint_solver import lambda_opt

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("from_ge/Curve_in_base_frame2.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	opt=lambda_opt(curve,curve_direction,steps=len(curve),robot1=abb6640(d=50))
	R_temp=opt.direction2R(curve_direction[0],-curve[1]+curve[0])
	q_init=opt.robot1.inv(curve[0],R_temp)[0]
	q_out=opt.single_arm_stepwise_optimize(q_init)

	###output to csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('from_ge/Curve_js2.csv',header=False,index=False)


if __name__ == "__main__":
	main()
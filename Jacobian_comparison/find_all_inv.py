import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robot_def import *

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
	data = read_csv("curve_poses/curve_pose0/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


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
	
	curve_js_all=[]

	###get initial all possible poses
	q_inits=np.array(inv(curve[0],curve_R[0]))

	for q_init in q_inits:
		curve_js_temp=[q_init]
		possible=True
		for i in range(len(curve)-1):
			q_all=np.array(inv(curve[i+1],curve_R[i+1]))
			if len(q_all)==0:
				possible=False
				break
			###choose inv_kin closest to previous joints
			temp_q=q_all-curve_js_temp[-1]
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			q_next=q_all[order[0]]
			if np.linalg.norm(q_next-curve_js_temp[-1])>0.2:
				#if next joint config too far from previous
				possible=False
				break
			curve_js_temp.append(q_all[order[0]])

		if possible:
			curve_js_all.append(curve_js_temp)
	print("possible paths: ",len(curve_js_all))





	###output to csv
	for i in range(len(curve_js_all)):
		curve_js=np.array(curve_js_all[i])
		df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
		df.to_csv('curve_poses/curve_pose0/Curve_backproj_js'+str(i)+'.csv',header=False,index=False)




if __name__ == "__main__":
	main()
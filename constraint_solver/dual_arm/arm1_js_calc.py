import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
sys.path.append('../toolbox')
from robot_def import *

def format_movej(q):

	eax='[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]'
	speed='v5000'
	zone='fine'
	q_deg=np.degrees(q)
	return 'MoveAbsJ '+'[['+str(q_deg[0])+','+str(q_deg[1])+','+str(q_deg[2])+','+str(q_deg[3])+','+str(q_deg[4])+','+str(q_deg[5])+'],'+eax+'],'\
			+speed+','+zone+',Paintgun;'


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

	###second arm settings
	arm2_base_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
	arm2_base_p=np.array([6000,0,0])

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("curve_poses/dual_arm/arm2_cs.csv", names=col_names)
	curve_arm2_x=data['X'].tolist()
	curve_arm2_y=data['Y'].tolist()
	curve_arm2_z=data['Z'].tolist()

	curve_arm2=np.vstack((curve_arm2_x, curve_arm2_y, curve_arm2_z)).T


	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("curve_poses/dual_arm/relative_path.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve_relative=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


	curve_js=np.zeros((len(curve_relative),6))

	q_init=np.array([0.627463700138299,0.17976842821744082,0.5196590573281621,1.6053098733278601,-0.8935105128511388,0.9174696574156079])
	position_prev=None
	for i in range(len(curve_relative)):
		try:
			position=np.dot(arm2_base_R,curve_arm2[i])+arm2_base_p+curve_relative[i]
			if i==0:
				position_next=np.dot(arm2_base_R,curve_arm2[1])+arm2_base_p+curve_relative[1]
				R=direction2R(curve_direction[i],-position_next+position)
			else:
				R=direction2R(curve_direction[i],-position+position_prev)
			q_all=np.array(inv(position,R))
			position_prev=position
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
				print(position)
				break
				pass

	###output to csv
	df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
	df.to_csv('curve_poses/dual_arm/arm1_js.csv',header=False,index=False)

	for i in range(0,len(curve_js),4999):
		print(format_movej(curve_js[i]))



if __name__ == "__main__":
	main()
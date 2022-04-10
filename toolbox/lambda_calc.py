import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from robots_def import *

def calc_lam_js(curve_js,robot):
	#calculate lambda from joints
	lam=[0]
	curve=[]
	for i in range(len(curve_js)):
	    robot_pose=robot.fwd(curve_js[i])
	    curve.append(robot_pose.p)
	    if i>0:
	        lam.append(lam[-1]+np.linalg.norm(curve[i]-curve[i-1]))
	return np.array(lam)

def calc_lam_cs(curve):
	###find path length
	lam=[0]
	for i in range(len(curve)-1):
		lam.append(lam[-1]+np.linalg.norm(curve[i+1]-curve[i]))

	return np.array(lam)

def calc_lamdot(curve_js,lam,robot,step):
	############find maximum lambda dot vs lambda
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###robot: joint velocity & acc limit 
	###step: step size used for dense curve

	dlam_max1=[]
	dlam_max2=[]

	dqdlam=[]
	d2qdlam2=[]

	for i in range(0,len(lam)-step,step):
		dq=curve_js[i+step]-curve_js[i]
		dqdlam.append(dq/(lam[i+step]-lam[i]))

		dlam_max1.append(np.min(np.divide(robot.joint_vel_limit,np.abs(dqdlam[-1]))))
		if i>0:
			d2qdlam2.append(2*(dqdlam[-1]-dqdlam[-2])/(lam[i+step]-lam[i-step]))
			dlam_max2.append(np.sqrt(np.min(np.divide(robot.joint_acc_limit,np.abs(d2qdlam2[-1])))))

	dlam_max_act=np.minimum(np.array(dlam_max1),np.insert(dlam_max2,0,99999))

	return dlam_max_act

def est_lamdot(dqdlam_list,breakpoints,lam,spl_list,merged_idx,robot):
	############estimated lambdadot from arbitray blending
	###dqdlam_list: list of dqdlam in each segment
	###lam: discrete lambda (path length), same shape as curve_js
	###breakpoints: breakpoints
	###spl_list: spline coefficient at each breakpoint
	###merged_idx: merged breakpoitn idx when blending
	###robot: robot definitions

	dlam_max1=[]
	dlam_max2=[]
	for dqdlam_seg in dqdlam_list:
		dlam_max1.append(np.min(np.divide(robot.joint_vel_limit,np.abs(dqdlam_seg))))

	###loop all spline blending
	for i in range(len(spl_list)):
		###loop all merged blending breakpoint
		if len(merged_idx[i])>1:
			bp=int(np.average(merged_idx[i]))
		else:
			bp=merged_idx[i][0]
		dq2dlam2=[]
		###loop all joints
		for j in range(len(robot.joint_vel_limit)):
			dderivative=spl_list[i][j].derivative().derivative()
			sampling=np.linspace(lam[bp]-5,lam[bp]+5)
			dq2dlam2.append(np.max(dderivative(lam[bp])))

		###calc lamdot_min at breakpoints
		dlam_max2.append(np.sqrt(np.min(np.divide(robot.joint_acc_limit,np.abs(dq2dlam2)))))
	return dlam_max1, dlam_max2
def est_lamdot_min(dqdlam_list,breakpoints,lam,spl_list,merged_idx,robot):
	dlam_max1,dlam_max2=est_lamdot(dqdlam_list,breakpoints,lam,spl_list,merged_idx,robot)
	# print(dlam_max2)
	return min(np.min(dlam_max1),np.min(dlam_max2))

def calc_lamdot2(curve_js,lam,robot,step):
	############find maximum lambda dot vs lambda, with different downsampling strategy, require full dense curve
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###robot: joint velocity & acc limit 
	###step: step size used for dense curve

	dlam_max1=[]
	dlam_max2=[]

	dqdlam=[]
	d2qdlam2=[]

	for i in range(1,len(lam)-step,step):
		dq=curve_js[i+1]-curve_js[i]
		dqdlam=dq/(lam[i+1]-lam[i])
		dlam_max1.append(np.min(np.divide(robot.joint_vel_limit,np.abs(dqdlam[-1]))))

		d2qdlam2.append(2*(dqdlam[-1]-dqdlam[-2])/(lam[i+1]-lam[i-1]))

		dlam_max2.append(np.sqrt(np.min(np.divide(robot.joint_acc_limit,np.abs(d2qdlam2[-1])))))



	dlam_max_act=np.minimum(np.array(dlam_max1),np.array(dlam_max2))


	return dlam_max1,dlam_max2

def calc_lamdot_acc_constraints(curve_js,lam,joint_vel_limit,joint_acc_limit,breakpoints,step):
	############find maximum lambda dot vs lambda
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###joint_vel_limit: joint velocity limit in radians

	dlam_max=[]

	joint_vel_prev=np.zeros(6)
	num_steps=int(len(curve_js)/step)
	idx=[]
	for i in range(len(breakpoints)-1):
		for j in range(breakpoints[i],1+breakpoints[i]+step*int((breakpoints[i+1]-breakpoints[i])/step),step):
			idx.append(j)

			next_step=min(breakpoints[i+1],j+step)

			dq=np.abs(curve_js[next_step]-curve_js[j])
			dqdlam=dq/(lam[next_step]-lam[j])
			t=np.max(np.divide(dq,joint_vel_limit))
			qdot_max=dq/t 		###approximated max qdot
			if j in breakpoints or next_step in breakpoints:
				t_acc=np.max(np.divide(joint_vel_prev,joint_acc_limit))
				if np.linalg.norm(dq-t_acc*qdot_max/2)>0:
					t_rem=np.max(np.divide(dq-t_acc*qdot_max/2,joint_vel_limit))
					qdot_act=dq/(t_acc+t_rem)
					dlam_max.append(qdot_act[0]/dqdlam[0])
					joint_vel_prev=qdot_max
				else:
					qddot_max=qdot_max/t_acc
					t_act=np.sqrt(2*dq/qddot_max)[0]
					dlam_max.append((qdot_max[0]/2)/dqdlam[0])
					joint_vel_prev=qddot_max*t_act
			else:
				dlam_max.append(qdot_max[0]/dqdlam[0])

	return dlam_max,idx


def calc_lamdot_dual(curve_js1,curve_js2,lam,joint_vel_limit1,joint_vel_limit2,step):
	############find maximum lambda dot vs lambda
	###curve_js1,2: curve expressed in joint space in radians for both robots
	###lam: discrete lambda (relative path length), same shape as curve_js, from 0 to 1
	###joint_vel_limit1,2: joint velocity limit in radians for both robots
	###step: step size used for dense curve

	dlam_max=[]

	for i in range(0,len(lam)-step,step):
		dq=np.abs(curve_js[i+step]-curve_js[i])
		dqdlam=dq/(lam[i+step]-lam[i])
		t=np.max(np.divide(dq,joint_vel_limit))

		qdot_max=dq/t 		###approximated max qdot
		dlam_max.append(qdot_max[0]/dqdlam[0])



	return dlam_max


def main():
	# col_names=['x', 'y', 'z','R1','R2','R3','R4','R5','R6','R7','R8','R9'] 
	# data = read_csv("../greedy_fitting/curve_fit_backproj.csv")
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	# print(curve)

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_ge/Curve_js2.csv", names=col_names)
	# data = read_csv("../constraint_solver/single_arm/trajectory/all_theta_opt/all_theta_opt_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	lam=calc_lam_cs(curve)
	robot=abb6640(d=50)
	step=1000
	lam_dot=calc_lamdot(curve_js,lam,robot,step)
	plt.plot(lam[::step][1:],lam_dot)
	plt.xlabel('path length (mm)')
	plt.ylabel('max lambda_dot')
	plt.title('lambda_dot vs lambda')
	plt.show()

if __name__ == "__main__":
	main()
import numpy as np
import scipy, copy
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from robots_def import *
from utils import * 
from scipy.optimize import fminbound

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

def calc_lam_js_2arm(curve_js1,curve_js2,robot1,robot2,base2_R,base2_p):
	curve=[]
	for i in range(len(curve_js1)):
		pose1_now=robot1.fwd(curve_js1[i])
		pose2_world_now=robot2.fwd(curve_js2[i],base2_R,base2_p)

		curve.append(pose2_world_now.R.T@(pose1_now.p-pose2_world_now.p))
	# visualize_curve(np.array(curve))
	lam=calc_lam_cs(curve)
	return np.array(lam)

def calc_lam_cs(curve):
	###find path length
	temp=np.diff(curve,axis=0)
	temp=np.linalg.norm(temp,axis=1)
	lam=np.insert(np.cumsum(temp),0,0)

	return lam

def calc_lamdot(curve_js,lam,robot,step):
	############find maximum lambda dot vs lambda
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###robot: joint velocity & acc limit 
	###step: step size used for dense curve

	# dlam_max1=[]
	# dlam_max2=[]

	# dqdlam=[]
	# d2qdlam2=[]

	# for i in range(0,len(lam)-step,step):
	# 	dq=curve_js[i+step]-curve_js[i]
	# 	dqdlam.append(dq/(lam[i+step]-lam[i]))

	# 	dlam_max1.append(np.min(np.divide(robot.joint_vel_limit,np.abs(dqdlam[-1]))))
	# 	if i>0:
	# 		d2qdlam2.append(2*(dqdlam[-1]-dqdlam[-2])/(lam[i+step]-lam[i-step]))
	# 		dlam_max2.append(np.sqrt(np.min(np.divide(robot.joint_acc_limit,np.abs(d2qdlam2[-1])))))

	# dlam_max_act=np.minimum(np.array(dlam_max1),np.insert(dlam_max2,0,99999))

	curve_js=curve_js[::step]
	lam=lam[::step]
	dq=np.gradient(curve_js,axis=0)
	dlam=np.gradient(lam)
	dqdlam=np.divide(dq.T,dlam).T

	d2qdlam2=np.divide(np.gradient(dqdlam,axis=0).T,dlam).T

	dlam_max1=np.min(np.divide(robot.joint_vel_limit,np.abs(dqdlam)),axis=1)
	dlam_max2=np.sqrt(np.min(np.divide(robot.joint_acc_limit,np.abs(d2qdlam2)),axis=1))

	dlam_max_act=np.minimum(dlam_max1,dlam_max2)

	return dlam_max_act
def calc_lamdot_2arm(curve_js,lam,robot1,robot2,step):
	############find maximum lambda dot vs lambda
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###robot: joint velocity & acc limit 
	###step: step size used for dense curve

	curve_js=curve_js[::step]
	lam=lam[::step]
	dq=np.gradient(curve_js,axis=0)
	dlam=np.gradient(lam)
	dqdlam=np.divide(dq.T,dlam).T

	d2qdlam2=np.divide(np.gradient(dqdlam,axis=0).T,dlam).T

	dlam_max1=np.min(np.divide(np.hstack((robot1.joint_vel_limit,robot2.joint_vel_limit)),np.abs(dqdlam)),axis=1)
	dlam_max2=np.sqrt(np.min(np.divide(np.hstack((robot1.joint_acc_limit,robot2.joint_acc_limit)),np.abs(d2qdlam2)),axis=1))

	dlam_max_act=np.minimum(dlam_max1,dlam_max2)

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

def traj_speed_est(robot,curve_js,lam,vd):
	###find desired qdot at each step
	dq=np.gradient(curve_js,axis=0)
	dlam=np.gradient(lam)
	dt=dlam/vd
	qdot_d=np.divide(dq,np.tile(np.array([dt]).T,(1,6)))
	###bound desired qdot with qdot constraint
	qdot_max=np.tile(robot.joint_vel_limit,(len(curve_js),1))
	coeff=np.divide(np.abs(qdot_d),qdot_max)
	coeff=np.max(coeff,axis=1)	###get the limiting joint
	coeff=np.clip(coeff,1,999)	###clip coeff to 1
	coeff=np.tile(np.array([coeff]).T,(1,6))

	qdot=np.divide(qdot_d,coeff)	###clip propotionally
		



	# ###iterate a few times to satisfy qddot constraint
	for r in range(100):
		dqdot=np.gradient(qdot,axis=0)

		qddot_d=np.divide(dqdot,np.tile(np.array([dt]).T,(1,6)))
		###bound desired qddot with qddot constraint
		qddot_max=np.tile(robot.joint_acc_limit,(len(curve_js),1))


		coeff=np.divide(np.abs(qddot_d),qddot_max)
		coeff=np.max(coeff,axis=1)	###get the limiting joint

		# qddot_lim_violate_idx=np.argwhere(coeff>1)
		# qdot[qddot_lim_violate_idx-1]*=0.999

		coeff=np.clip(coeff,1,999)	###clip coeff to 1

		coeff=np.tile(np.array([coeff]).T,(1,6))
		qddot=np.divide(qddot_d,coeff)	###clip propotionally

		qdot[1:]=qdot[:-1]+qddot[:-1]*np.tile(np.array([dt[:-1]]).T,(1,6))



		###update dt
		dt=np.max(dq/qdot,axis=1)

	speed=[]
	for i in range(len(curve_js)):
		J=robot.jacobian(curve_js[i])
		speed.append(np.linalg.norm((J@qdot[i])[3:]))

	return speed

def q_linesearch(alpha,qdot_prev,qdot_next,dt,joint_acc_limit):
	###alpha: coefficient of next qdot, (0,1]
	qddot=(alpha*qdot_next-qdot_prev)/dt
	coeff=np.abs(qddot)/joint_acc_limit

	###if find one alpha within acc constraint, take it
	if np.max(coeff)<1:
		return -alpha
	###else choose one alpha that brings outcome qddot closest to acc constraint
	else:
		return np.abs(np.max(coeff)-1)

def traj_speed_est2(robot,curve_js,lam,vd):

	###find desired qdot at each step
	dq=np.gradient(curve_js,axis=0)
	dlam=np.gradient(lam)
	dt=dlam/vd
	qdot_d=np.divide(dq,np.tile(np.array([dt]).T,(1,6)))
	###bound desired qdot with qdot constraint
	qdot_max=np.tile(robot.joint_vel_limit,(len(curve_js),1))
	coeff=np.divide(np.abs(qdot_d),qdot_max)
	coeff=np.max(coeff,axis=1)	###get the limiting joint
	coeff=np.clip(coeff,1,999)	###clip coeff to 1
	coeff=np.tile(np.array([coeff]).T,(1,6))

	qdot=np.divide(qdot_d,coeff)	###clip propotionally
	
	#traversal
	qdot_act=[qdot[0]]
	alpha_all=[]
	for i in range(1,len(curve_js)):

		alpha=fminbound(q_linesearch,0,1.,args=(qdot_act[-1],qdot[i],dt[i],robot.joint_acc_limit))
		line_out=q_linesearch(alpha,qdot_act[-1],qdot[i],dt[i],robot.joint_acc_limit)

		qdot_act.append(alpha*qdot[i])
		alpha_all.append(alpha)

	speed=[]
	for i in range(len(curve_js)):
		J=robot.jacobian(curve_js[i])
		speed.append(np.linalg.norm((J@qdot_act[i])[3:]))

	return speed

def main():
	robot=abb6640(d=50)
	curve_js = read_csv("../data/wood/Curve_js.csv",header=None).values

	# train_data = read_csv("../constraint_solver/single_arm/trajectory/curve_pose_opt/arm1.csv", names=col_names)
	# train_data = read_csv("qsol.csv", names=col_names)
	# train_data = read_csv("../constraint_solver/single_arm/trajectory/all_theta_opt/all_theta_opt_js.csv", names=col_names)

	lam=calc_lam_js(curve_js,robot)
	
	step=10
	lam_dot=calc_lamdot(curve_js,lam,robot,step)
	plt.plot(lam[::step],lam_dot)
	plt.xlabel('path length (mm)')
	plt.ylabel('max lambda_dot')
	plt.ylim([0,4000])
	plt.title('lambda_dot vs lambda')
	plt.show()

def main2():
	###read in points
	curve_js1 = read_csv("../constraint_solver/dual_arm/trajectory/arm1.csv",header=None).values
	curve_js2 = read_csv("../constraint_solver/dual_arm/trajectory/arm2.csv",header=None).values
	###define robots
	robot1=abb1200(d=50)
	robot2=abb6640()

	###read in robot2 pose
	with open('../constraint_solver/dual_arm/trajectory/abb6640.yaml') as file:
		H_6640 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_6640[:-1,:-1]
	base2_p=1000.*H_6640[:-1,-1]
	
	lam=calc_lam_js_2arm(curve_js1,curve_js2,robot1,robot2,base2_R,base2_p)
	step=10
	lam_dot=calc_lamdot_2arm(np.hstack((curve_js1,curve_js2)),lam,robot1,robot2,step)
	plt.plot(lam[::step],lam_dot)
	plt.xlabel('path length (mm)')
	plt.ylabel('max lambda_dot')
	plt.title('lambda_dot vs lambda')
	plt.ylim([0,2000])
	plt.show()

def main3():
	from MotionSend import MotionSend
	from blending import form_traj_from_bp,blend_js_from_primitive

	robot=abb6640(d=50)
	# curve_js = read_csv("../data/wood/baseline/100L/curve_fit_js.csv",header=None).values
	# curve_js = read_csv("../data/wood/Curve_js.csv",header=None).values

	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('../data/from_NX/baseline/100L/command.csv')
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('../ILC/max_gradient/curve1_250_100L_multipeak/command.csv')
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('../ILC/max_gradient/curve2_1100_100L_multipeak/command.csv')
	curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)
	curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

	lam=calc_lam_js(curve_js_blended,robot)
	speed=traj_speed_est2(robot,curve_js_blended,lam,1200)
	plt.plot(lam,speed)
	plt.show()

if __name__ == "__main__":
	main3()
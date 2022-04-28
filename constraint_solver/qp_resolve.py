import numpy as np
from pandas import *
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
sys.path.append('../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
from error_check import *

def single_arm_stepwise_optimize(robot,q_init,lam,lamdot_des,curve,curve_normal):

		q_all=[q_init]
		lam_out=[0]
		curve_out=[curve[0]]
		curve_normal_out=[curve_normal[0]]
		ts=0.01
		total_time=lam[-1]/lamdot_des
		total_timestep=int(total_time/ts)
		idx=0
		lam_qp=0
		K_trak_p=100
		K_trak_R=100
		qdot_prev=[]
		act_speed=[]
		# for i in range(1,total_timestep):
		while lam_qp<lam[-1] and idx<len(curve)-1:
			#find corresponding idx in curve & curve normal
			prev_idx=np.abs(lam-lam_qp).argmin()
			idx=np.abs(lam-(lam_qp+ts*lamdot_des)).argmin()

			pose_now=robot.fwd(q_all[-1])
			
			Kw=10
			Kq=.01*np.eye(6)    #small value to make sure positive definite
			KR=np.eye(3)        #gains for position and orientation error

			J=robot.jacobian(q_all[-1])        #calculate current Jacobian
			Jp=J[3:,:]
			JR=J[:3,:]
			JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

			H=np.dot(np.transpose(Jp),Jp)+Kq+Kw*np.dot(np.transpose(JR_mod),JR_mod)
			H=(H+np.transpose(H))/2
			#####vd = - v_des - K ( x - x_des)
			v_des=(curve[idx]-curve[prev_idx])/ts
			vd=v_des+K_trak_p*(curve[prev_idx]-pose_now.p)
			ezdot_des=(curve_normal[idx]-curve_normal[prev_idx])/ts
			ezdotd=ezdot_des+K_trak_R*(curve_normal[prev_idx]-pose_now.R[:,-1])

			#####vd = K ( x_des_next - x_cur)
			# vd=(curve[idx]-pose_now.p)/ts
			# vd=lamdot_des*vd/np.linalg.norm(vd)
			# ezdotd=(curve_normal[idx]-pose_now.R[:,-1])/ts

			print(np.linalg.norm(vd))

			f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),ezdotd)
			if len(qdot_prev)==0:
				lb=-robot.joint_vel_limit
				ub=robot.joint_vel_limit
			else:
				lb=np.maximum(-robot.joint_vel_limit,(qdot_prev-robot.joint_acc_limit)*ts)
				ub=np.minimum(robot.joint_vel_limit,(qdot_prev+robot.joint_acc_limit)*ts)

			qdot=solve_qp(H,f,lb=lb,ub=ub)
			qdot_prev=qdot

			q_all.append(q_all[-1]+ts*qdot)

			curve_out.append(pose_now.p)
			curve_normal_out.append(pose_now.R[:,-1])

			
			lam_qp+=np.linalg.norm(robot.fwd(q_all[-1]).p-pose_now.p)
			lam_out.append(lam_qp)

			act_speed.append(np.linalg.norm(np.dot(Jp,qdot)))

		q_all=np.array(q_all)
		curve_out=np.array(curve_out)
		curve_normal_out=np.array(curve_normal_out)

		return q_all,lam_out,curve_out,curve_normal_out,act_speed


# def dual_arm_stepwise_optimize(robot1,robot2,q_init1,q_init2,base1_p,base1_R,base2_p,base2_R,lam,lamdot_des,curve,curve_normal):
# 		#curve_normal: expressed in second robot tool frame
# 		q1_all=[q_init1]
# 		q2_all=[q_init2]

# 		lam_out=[0]
# 		curve_relative_out=[np.zeros(3)]
# 		curve_relative_normal_out=[curve_normal[0]]
# 		ts=0.01
# 		total_time=lam[-1]/lamdot_des
# 		total_timestep=int(total_time/ts)
# 		idx=0
# 		lam_qp=0
# 		K_trak=100
# 		Kw=1000
# 		q1dot_prev=[]
# 		q2dot_prev=[]
# 		act_speed=[]

# 		while lam_qp<lam[-1] and idx<len(curve)-1:

# 			#find corresponding idx in curve & curve normal
# 			prev_idx=np.abs(lam-lam_qp).argmin()
# 			idx=np.abs(lam-(lam_qp+ts*lamdot_des)).argmin()

# 			pose1_now=robot1.fwd(q1_all[-1])
# 			pose2_now=robot2.fwd(q2_all[-1])

# 			pose2_world_now=robot2.fwd(q2_all[-1],base2_R,base2_p)

# 			###error indicator
# 			error=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-curve[idx])
# 			ori_error=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-curve_normal[idx])	
# 			print(lam_qp,error,ori_error)
# 			########################################################first robot###########################################
# 			Kq=.01*np.eye(12)    #small value to make sure positive definite

# 			J1=robot1.jacobian(q1_all[-1])        #calculate current Jacobian
# 			J1p=J1[3:,:]
# 			J1R=J1[:3,:] 
# 			J1R_mod=-np.dot(hat(pose1_now.R[:,-1]),J1R)


# 			#####vd = - v_des - K ( x - x_des)
# 			# v1_des=np.dot(pose2_world_now.R,curve[idx]-curve[prev_idx])/ts
# 			# v1d=v1_des+K_trak*np.dot(pose2_world_now.R,(curve[prev_idx]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)))

# 			# ezdot1_des=np.dot(pose2_world_now.R,curve_normal[idx]-curve_normal[prev_idx])/ts
# 			# ezdotd1=ezdot1_des+K_trak*np.dot(pose2_world_now.R,curve_normal[prev_idx]-np.dot(pose2_world_now.R.T,pose1_now.R)[:,-1])

# 			# k=np.dot(pose2_world_now.R,np.cross(curve_normal[prev_idx],curve_normal[idx]))	###R^2_0*(R^0_2*R1[z] x curve_normal)
# 			# k=k/np.linalg.norm(k)
# 			# theta=-np.arctan2(np.linalg.norm(np.cross(curve_normal[prev_idx],curve_normal[idx])),np.dot(curve_normal[prev_idx],curve_normal[idx]))
# 			# k=np.array(k)
# 			# s=np.sin(theta/2)*k         #eR2
# 			# w1_des=-s

# 			# k=np.dot(pose2_world_now.R,np.cross(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1]),curve_normal[prev_idx]))	###R^2_0*(R^0_2*R1[z] x curve_normal)
# 			# k=k/np.linalg.norm(k)
# 			# vec1=np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
# 			# theta=-np.arctan2(np.linalg.norm(np.cross(vec1,curve_normal[prev_idx])),np.dot(vec1,curve_normal[prev_idx]))
# 			# k=np.array(k)
# 			# s=np.sin(theta/2)*k         #eR2
# 			# w1d=w1_des-K_trak*s

# 			#####vd = K ( x_des_next - x_cur)
# 			v1d=np.dot(pose2_world_now.R,(curve[idx]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)))/ts
# 			ezdotd1=np.dot(pose2_world_now.R,curve_normal[idx]-np.dot(pose2_world_now.R.T,pose1_now.R)[:,-1])/ts
# 			# ezdotd1=(np.dot(pose2_world_now.R,curve_normal[idx])-pose1_now.R[:,-1])/ts

# 			vec1=np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
# 			k=np.dot(pose2_world_now.R,np.cross(vec1,curve_normal[idx]))	###R^2_0*(R^0_2*R1[z] x curve_normal)
# 			k=k/np.linalg.norm(k)
# 			theta=-np.arctan2(np.linalg.norm(np.cross(vec1,curve_normal[idx])),np.dot(vec1,curve_normal[idx]))
# 			k=np.array(k)
# 			s=np.sin(theta/2)*k         #eR2
# 			w1d=-s

# 			########################################################Second robot###########################################
# 			J2=robot2.jacobian(q2_all[-1])        #calculate current Jacobian, mapped to robot1 base frame
# 			J2p=np.dot(base2_R,J2[3:,:])
# 			J2R=np.dot(base2_R,J2[:3,:])
# 			J2R_mod=-np.dot(hat(pose2_now.R[:,-1]),J2[:3,:])
# 			J2R_mod=np.dot(base2_R,J2R_mod)

# 			J_all=np.zeros((12,12))
# 			J_all[3:6,:6]=J1p
# 			J_all[:3,:6]=J1R#_mod
# 			J_all[9:,6:]=J2p
# 			J_all[6:9,6:]=J2R#_mod


# 			J_all_p=np.vstack((J_all[3:6,:],J_all[9:,:]))
# 			J_all_R=np.vstack((J_all[:3,:],J_all[6:9,:]))
			
# 			H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
# 			H=(H+np.transpose(H))/2
			


# 			nu_d=np.hstack((ezdotd1,v1d,-ezdotd1,-v1d))

# 			vd=np.hstack((v1d,-v1d))
# 			ezdotd=np.hstack((ezdotd1,-ezdotd1))
# 			wd=np.hstack((w1d,-w1d))

# 			# f=-np.dot(np.transpose(J_all),nu_d)
# 			f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),wd)
# 			# f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)

# 			if len(q2dot_prev)==0:
# 				lb=np.hstack((-robot1.joint_vel_limit,-robot2.joint_vel_limit))
# 				ub=-lb
# 			else:
# 				# lb=np.maximum(-robot2.joint_vel_limit,(q2dot_prev-robot1.joint_acc_limit)*ts)
# 				# ub=np.minimum(robot2.joint_vel_limit,(q2dot_prev+robot1.joint_acc_limit)*ts)
# 				lb=np.hstack((-robot1.joint_vel_limit,-robot2.joint_vel_limit))
# 				ub=-lb

# 			qdot=solve_qp(H,f,lb=lb,ub=ub)
# 			# print(qdot)

# 			q1dot_prev=qdot[:6]
# 			q2dot_prev=qdot[6:]
# 			q1_all.append(q1_all[-1]+ts*qdot[:6])
# 			q2_all.append(q2_all[-1]+ts*qdot[6:])

# 			dist_now=np.dot(pose2_world_now.R.T,robot1.fwd(q1_all[-1]).p-robot2.fwd(q2_all[-1],base2_R,base2_p).p)
# 			dist_prev=np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)
# 			lam_qp+=np.linalg.norm(dist_now - dist_prev)
# 			lam_out.append(lam_qp)
# 			curve_relative_out.append(dist_now)
# 			curve_relative_normal_out.append(np.dot(robot2.fwd(q2_all[-1],base2_R,base2_p).R.T,robot1.fwd(q1_all[-1]).R)[:,-1])

# 		q1_all=np.array(q1_all)
# 		q2_all=np.array(q2_all)
# 		curve_relative_out=np.array(curve_relative_out)
# 		curve_relative_normal_out=np.array(curve_relative_normal_out)

# 		return q1_all, q2_all,	lam_out, curve_relative_out, curve_relative_normal_out, []

def dual_arm_stepwise_optimize(robot1,robot2,q_init1,q_init2,base1_p,base1_R,base2_p,base2_R,lam,lamdot_des,curve,curve_normal):
		#curve_normal: expressed in second robot tool frame
		q1_all=[q_init1]
		q2_all=[q_init2]

		lam_out=[0]
		curve_relative_out=[np.zeros(3)]
		curve_relative_normal_out=[curve_normal[0]]
		ts=0.01
		total_time=lam[-1]/lamdot_des
		total_timestep=int(total_time/ts)
		idx=0
		lam_qp=0
		K_trak=100
		Kw=100
		qdot_prev=[]

		act_speed=[]

		joint_vel_limit=np.hstack((robot1.joint_vel_limit,robot2.joint_vel_limit))
		upper_limit=np.hstack((robot1.upper_limit,robot2.upper_limit))
		lower_limit=np.hstack((robot1.lower_limit,robot2.lower_limit))
		joint_acc_limit=np.hstack((robot1.joint_acc_limit,robot2.joint_acc_limit))

		while lam_qp<lam[-1] and idx<len(curve)-1:

			#find corresponding idx in curve & curve normal
			prev_idx=np.abs(lam-lam_qp).argmin()
			idx=np.abs(lam-(lam_qp+ts*lamdot_des)).argmin()

			# print(prev_idx,idx)

			pose1_now=robot1.fwd(q1_all[-1])
			pose2_now=robot2.fwd(q2_all[-1])

			pose2_world_now=robot2.fwd(q2_all[-1],base2_R,base2_p)

			###error indicator
			# error=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-curve[idx])
			# ori_error=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-curve_normal[idx])	
			# print(lam_qp,error,ori_error)
			########################################################first robot###########################################
			Kq=.01*np.eye(12)    #small value to make sure positive definite

			J1=robot1.jacobian(q1_all[-1])        #calculate current Jacobian
			J1p=J1[3:,:]
			J1R=J1[:3,:] 
			J1R_mod=-np.dot(hat(pose1_now.R[:,-1]),J1R)


			#####vd = - v_des - K ( x - x_des)
			v1_des=np.dot(pose2_world_now.R,curve[idx]-curve[prev_idx])/ts
			v1d=v1_des+K_trak*np.dot(pose2_world_now.R,(curve[prev_idx]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)))

			ezdot1_des=np.dot(pose2_world_now.R,curve_normal[idx]-curve_normal[prev_idx])/ts
			ezdotd1=ezdot1_des+K_trak*np.dot(pose2_world_now.R,curve_normal[prev_idx]-np.dot(pose2_world_now.R.T,pose1_now.R)[:,-1])

			k=np.dot(pose2_world_now.R,np.cross(curve_normal[prev_idx],curve_normal[idx]))	###R^2_0*(R^0_2*R1[z] x curve_normal)
			if np.linalg.norm(k)==0:
				w1_des=np.zeros(3)
			else:
				k=k/np.linalg.norm(k)
				theta=-np.arctan2(np.linalg.norm(np.cross(curve_normal[prev_idx],curve_normal[idx])),np.dot(curve_normal[prev_idx],curve_normal[idx]))
				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				w1_des=-s

			k=np.dot(pose2_world_now.R,np.cross(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1]),curve_normal[prev_idx]))	###R^2_0*(R^0_2*R1[z] x curve_normal)
			if np.linalg.norm(k)==0:
				w1d=w1_des
			else:
				k=k/np.linalg.norm(k)
				vec1=np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
				theta=-np.arctan2(np.linalg.norm(np.cross(vec1,curve_normal[prev_idx])),np.dot(vec1,curve_normal[prev_idx]))
				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				w1d=w1_des-K_trak*s

			#####vd = K ( x_des_next - x_cur)
			# v1d=np.dot(pose2_world_now.R,(curve[idx]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)))/ts
			# ezdotd1=np.dot(pose2_world_now.R,curve_normal[idx]-np.dot(pose2_world_now.R.T,pose1_now.R)[:,-1])/ts
			# # ezdotd1=(np.dot(pose2_world_now.R,curve_normal[idx])-pose1_now.R[:,-1])/ts

			# vec1=np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
			# k=np.dot(pose2_world_now.R,np.cross(vec1,curve_normal[idx]))	###R^2_0*(R^0_2*R1[z] x curve_normal)
			# k=k/np.linalg.norm(k)
			# theta=-np.arctan2(np.linalg.norm(np.cross(vec1,curve_normal[idx])),np.dot(vec1,curve_normal[idx]))
			# k=np.array(k)
			# s=np.sin(theta/2)*k         #eR2
			# w1d=-s

			########################################################Second robot###########################################
			J2=robot2.jacobian(q2_all[-1])        #calculate current Jacobian, mapped to robot1 base frame
			J2p=np.dot(base2_R,J2[3:,:])
			J2R=np.dot(base2_R,J2[:3,:])
			J2R_mod=-np.dot(hat(pose2_now.R[:,-1]),J2[:3,:])
			J2R_mod=np.dot(base2_R,J2R_mod)

			J_all=np.zeros((6,12))
			J_all[3:6,:6]=J1p
			J_all[:3,:6]=J1R#_mod
			J_all[3:6,6:]=-J2p
			J_all[:3,6:]=-J2R#_mod


			J_all_p=J_all[3:,:]
			J_all_R=J_all[:3,:]
			
			H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
			H=(H+np.transpose(H))/2
			
			# print(v1d)

			nu_d=np.hstack((ezdotd1,v1d))


			# f=-np.dot(np.transpose(J_all),nu_d)
			f=-np.dot(np.transpose(J_all_p),v1d)-Kw*np.dot(np.transpose(J_all_R),w1d)
			# f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)

			if len(qdot_prev)==0:
				lb=np.maximum(-joint_vel_limit,(lower_limit-np.hstack((q1_all[-1],q2_all[-1])))/ts)
				ub=np.minimum(joint_vel_limit,(upper_limit-np.hstack((q1_all[-1],q2_all[-1])))/ts)
			else:
				# lb=np.maximum(-robot2.joint_vel_limit,(q2dot_prev-robot1.joint_acc_limit)*ts)
				# ub=np.minimum(robot2.joint_vel_limit,(q2dot_prev+robot1.joint_acc_limit)*ts)
				lb=np.maximum(-joint_vel_limit,(lower_limit-np.hstack((q1_all[-1],q2_all[-1])))/ts)
				ub=np.minimum(joint_vel_limit,(upper_limit-np.hstack((q1_all[-1],q2_all[-1])))/ts)

			qdot=solve_qp(H,f,lb=lb,ub=ub)
			# print(qdot)

			qdot_prev=qdot
			q1_all.append(q1_all[-1]+ts*qdot[:6])
			q2_all.append(q2_all[-1]+ts*qdot[6:])


			dist_now=np.dot(pose2_world_now.R.T,robot1.fwd(q1_all[-1]).p-robot2.fwd(q2_all[-1],base2_R,base2_p).p)
			dist_prev=np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)
			lam_qp+=np.linalg.norm(dist_now - dist_prev)
			lam_out.append(lam_qp)
			curve_relative_out.append(dist_now)
			curve_relative_normal_out.append(np.dot(robot2.fwd(q2_all[-1],base2_R,base2_p).R.T,robot1.fwd(q1_all[-1]).R)[:,-1])

		q1_all=np.array(q1_all)
		q2_all=np.array(q2_all)
		curve_relative_out=np.array(curve_relative_out)
		curve_relative_normal_out=np.array(curve_relative_normal_out)

		return q1_all, q2_all,	lam_out, curve_relative_out, curve_relative_normal_out, []

# def dual_arm_stepwise_optimize(robot1,robot2,q_init1,q_init2,base1_p,base1_R,base2_p,base2_R,lam,lamdot_des,curve,curve_normal):
# 		#curve_normal: expressed in second robot tool frame
# 		q1_all=[q_init1]
# 		q2_all=[q_init2]

# 		lam_out=[0]
# 		curve_relative_out=[np.zeros(3)]
# 		curve_relative_normal_out=[curve_normal[0]]
# 		ts=0.0001
# 		total_time=lam[-1]/lamdot_des
# 		total_timestep=int(total_time/ts)
# 		idx=0
# 		lam_qp=0
# 		K_trak=1000
# 		Kw=100
# 		q1dot_prev=[]
# 		q2dot_prev=[]
# 		act_speed=[]

# 		while lam_qp<lam[-1] and idx<len(curve)-1:

# 			#find corresponding idx in curve & curve normal
# 			prev_idx=np.abs(lam-lam_qp).argmin()
# 			idx=np.abs(lam-(lam_qp+ts*lamdot_des)).argmin()

# 			pose1_now=robot1.fwd(q1_all[-1])
# 			pose2_now=robot2.fwd(q2_all[-1])

# 			pose2_world_now=robot2.fwd(q2_all[-1],base2_R,base2_p)

# 			###error indicator
# 			error=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-curve[idx])
# 			ori_error=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-curve_normal[idx])	
# 			print(lam_qp,error,ori_error)
# 			########################################################first robot###########################################
# 			Kq=.01*np.eye(12)    #small value to make sure positive definite

# 			J1=robot1.jacobian(q1_all[-1])        #modify jacobian
# 			J1p=J1[3:,:]
# 			J1R=-np.dot(hat(pose1_now.R[:,-1]),J1[:3,:])
# 			J1[:3,:]=J1R

# 			J2=robot2.jacobian(q2_all[-1])
# 			J2p=J2[3:,:]
# 			J2R=-np.dot(hat(pose2_now.R[:,-1]),J2[:3,:])
# 			J2[:3,:]=J2R

# 			J_all=np.zeros((6,12))
# 			transform1=np.zeros((6,6))
# 			transform2=np.zeros((6,6))
# 			transform1[:3,:3]=pose2_world_now.R.T
# 			transform1[3:,3:]=pose2_world_now.R.T
# 			J_all[:,:6]=np.dot(transform1,J1)
# 			transform2[:3,:3]=pose2_now.R.T
# 			transform2[3:,3:]=pose2_now.R.T
# 			J_all[:,6:]=np.dot(transform2,J2)

# 			J_all_p=np.hstack((np.dot(pose2_world_now.R.T,J1p),np.dot(pose2_now.R.T,J2p)))
# 			J_all_R=J_all[:3,:]

# 			#####vd = - v_des - K ( x - x_des)
# 			# v_des=(curve[idx]-curve[prev_idx])/ts
# 			# vd=v_des+K_trak*(curve[prev_idx]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))

# 			# ezdot_des=(curve_normal[idx]-curve_normal[prev_idx])/ts
# 			# ezdotd=ezdot_des+K_trak*(curve_normal[prev_idx]-np.dot(pose2_world_now.R.T,pose1_now.R)[:,-1])

# 			#####vd = K ( x_des_next - x_cur)
# 			vd=(curve[idx]-np.dot(pose2_world_now.R.T, pose1_now.p-pose2_world_now.p))/ts
# 			ezdotd=(curve_normal[idx]-np.dot(pose2_world_now.R.T,pose1_now.R)[:,-1])/ts
# 			# ezdotd1=(np.dot(pose2_world_now.R,curve_normal[idx])-pose1_now.R[:,-1])/ts
			
			
# 			H=np.dot(np.transpose(J_all_p),J_all_p)+Kq#+Kw*np.dot(np.transpose(J_all_R),J_all_R)
# 			H=(H+np.transpose(H))/2
			

# 			nu_d=np.hstack((vd,np.zeros(3)))

# 			f=-np.dot(np.transpose(J_all_p),vd)#-Kw*np.dot(np.transpose(J_all_R),ezdotd)
# 			# f=-np.dot(np.transpose(J_all),nu_d)
# 			if len(q2dot_prev)==0:
# 				lb=np.hstack((-robot1.joint_vel_limit,-robot2.joint_vel_limit))
# 				ub=-lb
# 			else:
# 				# lb=np.maximum(-robot2.joint_vel_limit,(q2dot_prev-robot1.joint_acc_limit)*ts)
# 				# ub=np.minimum(robot2.joint_vel_limit,(q2dot_prev+robot1.joint_acc_limit)*ts)
# 				lb=np.hstack((-robot1.joint_vel_limit,-robot2.joint_vel_limit))
# 				ub=-lb

# 			qdot=solve_qp(H,f,lb=lb,ub=ub)
# 			# print(qdot)

# 			q1dot_prev=qdot[:6]
# 			q2dot_prev=qdot[6:]
# 			q1_all.append(q1_all[-1]+ts*qdot[:6])
# 			q2_all.append(q2_all[-1]+ts*qdot[6:])

# 			dist_now=np.dot(pose2_world_now.R.T,robot1.fwd(q1_all[-1]).p-robot2.fwd(q2_all[-1],base2_R,base2_p).p)
# 			dist_prev=np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)
# 			lam_qp+=np.linalg.norm(dist_now - dist_prev)
# 			lam_out.append(lam_qp)
# 			curve_relative_out.append(dist_now)
# 			curve_relative_normal_out.append(np.dot(robot2.fwd(q2_all[-1],base2_R,base2_p).R.T,robot1.fwd(q1_all[-1]).R)[:,-1])

# 		q1_all=np.array(q1_all)
# 		q2_all=np.array(q2_all)
# 		curve_relative_out=np.array(curve_relative_out)
# 		curve_relative_normal_out=np.array(curve_relative_normal_out)

# 		return q1_all, q2_all,	lam_out, curve_relative_out, curve_relative_normal_out, []

def main():

	robot=abb6640(d=50)

	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv('../data/from_ge/Curve_in_base_frame2.csv', names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv('../data/from_ge/Curve_js2.csv', names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	# col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	# data = read_csv('../data/from_ge/q50000.csv', names=col_names)
	# curve_q1=data['q1'].tolist()
	# curve_q2=data['q2'].tolist()
	# curve_q3=data['q3'].tolist()
	# curve_q4=data['q4'].tolist()
	# curve_q5=data['q5'].tolist()
	# curve_q6=data['q6'].tolist()
	# curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	# curve=[]
	# curve_normal=[]
	# for i in range(len(curve_js)):
	# 	pose_temp=robot.fwd(curve_js[i])
	# 	curve.append(pose_temp.p)
	# 	curve_normal.append(pose_temp.R[:,-1])
	# curve=np.array(curve)
	# curve_normal=np.array(curve_normal)

	lam=calc_lam_cs(curve)
	lamdot_des=800

	q_all,lam_out,curve_out,curve_normal_out,act_speed=single_arm_stepwise_optimize(robot,curve_js[0],lam,lamdot_des,curve,curve_normal)

	###plot results
	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='red')
	ax.plot3D(curve_out[:,0], curve_out[:,1], curve_out[:,2], c='green')
	# plt.show()

	print(calc_max_error_w_normal(curve_out[2:],curve,curve_normal_out[2:],curve_normal))
	# df=DataFrame({'j1':q_all[:,0],'j2':q_all[:,1],'j3':q_all[:,2],'j4':q_all[:,3],'j5':q_all[:,4],'j6':q_all[:,5]})
	# df.to_csv('curve_qp_js.csv',header=False,index=False)

	lamdot=calc_lamdot(q_all,lam_out,robot,1)

	plt.figure(1)
	plt.plot(lam_out[1:],act_speed)
	plt.title("TCP speed (jacobian) vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('speed (mm/s)')
	plt.xlabel('lambda (mm)')


	plt.figure(2)
	plt.plot(lam_out,lamdot, label='Lambda Dot')
	plt.title("lamdot vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('lamdot (mm/s)')
	plt.xlabel('lambda (mm)')
	plt.legend()
	plt.show()

def main2():

	robot1=abb1200(d=50)
	robot2=abb6640()

	###read in points
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("../data/from_ge/relative_path_tool_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	relative_path=np.vstack((curve_x, curve_y, curve_z)).T
	relative_path_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	
	base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
	base2_p=np.array([3000,1000,0])

	#################################################calculate initial joint angles through inv#########################

	q_init1=np.array([1.041163124,	0.163121569,	0.060872101,	0.975429159,	-1.159317104,	-1.184301005])
	q_init2=np.array([0.141740433,	0.631713663,	0.458836251,	-0.321985253,	-1.545055101,	0.139048988])


	lam=calc_lam_cs(relative_path)

	lamdot_des=5000

	q1_all,q2_all,lam_out,curve_relative_out,curve_relative_normal_out,act_speed=dual_arm_stepwise_optimize(robot1,robot2,q_init1,q_init2,np.zeros(3),np.eye(3),base2_p,base2_R,lam,lamdot_des
		,relative_path,relative_path_direction)

	###plot results
	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot3D(relative_path[:,0], relative_path[:,1], relative_path[:,2], c='red')
	ax.plot3D(curve_relative_out[:,0], curve_relative_out[:,1], curve_relative_out[:,2], c='green')
	plt.show()

	print(calc_max_error_w_normal(curve_relative_out[2:],relative_path,curve_relative_normal_out[2:],relative_path_direction))
	df=DataFrame({'j1':q1_all[:,0],'j2':q1_all[:,1],'j3':q1_all[:,2],'j4':q1_all[:,3],'j5':q1_all[:,4],'j6':q1_all[:,5]})
	df.to_csv('dual_arm/trajectory/qp_arm1.csv',header=False,index=False)
	df=DataFrame({'j1':q2_all[:,0],'j2':q2_all[:,1],'j3':q2_all[:,2],'j4':q2_all[:,3],'j5':q2_all[:,4],'j6':q2_all[:,5]})
	df.to_csv('dual_arm/trajectory/qp_arm2.csv',header=False,index=False)

	lamdot=calc_lamdot_2arm(np.hstack((q1_all,q2_all)),lam_out,robot1,robot2,step=1)

	# plt.figure(1)
	# plt.plot(lam_out[1:],act_speed)
	# plt.title("TCP speed (jacobian) vs lambda, v_des= "+str(lamdot_des))
	# plt.ylabel('speed (mm/s)')
	# plt.xlabel('lambda (mm)')


	plt.figure(2)
	plt.plot(lam_out,lamdot, label='Lambda Dot')
	plt.title("lamdot vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('lamdot (mm/s)')
	plt.xlabel('lambda (mm)')
	plt.legend()
	plt.show()
if __name__ == '__main__':
	main2()
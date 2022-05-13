import numpy as np
from pandas import *
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, shgo, NonlinearConstraint, minimize, fminbound
from qpsolvers import solve_qp

sys.path.append('../../circular_Fit')

sys.path.append('../../toolbox')
from robots_def import *
from lambda_calc import *
from blending import *
from utils import *

class lambda_opt(object):
	###robot1 hold paint gun for single arm
	###robot1 hold paint gun, robot2 hold blade for dual arm
	def __init__(self,curve,curve_normal,robot1=abb6640(),robot2=abb1200(),base2_R=np.eye(3),base2_p=np.zeros(3),steps=32,breakpoints=[],primitives=[]):

		self.curve_original=curve
		self.curve_normal_original=curve_normal
		self.robot1=robot1
		self.robot2=robot2
		self.base2_R=base2_R
		self.base2_p=base2_p
		self.joint_vel_limit=np.radians([110,90,90,150,120,235])
		self.steps=steps

		self.lim_factor=0.0000001###avoid fwd error on joint limit

		#prespecified primitives
		self.primitives=primitives
		if len(breakpoints)>0:
			self.act_breakpoints=breakpoints
			self.act_breakpoints[1:]=self.act_breakpoints[1:]-1
			self.curve=curve[self.act_breakpoints]
			self.curve_normal=curve_normal[self.act_breakpoints]
			self.breakpoints=breakpoints
		else:
			###decrease curve density to simplify computation
			self.num_per_step=int(len(curve)/steps)	
			self.breakpoints=np.linspace(0,len(curve_normal),steps).astype(int)
			self.act_breakpoints=copy.deepcopy(self.breakpoints)
			self.act_breakpoints[1:]=self.breakpoints[1:]-1
			self.curve=curve[self.act_breakpoints]
			self.curve_normal=curve_normal[self.act_breakpoints]

		###find path length
		self.lam_original=calc_lam_cs(curve)
		self.lam=self.lam_original[self.act_breakpoints]

	def normalize_dq(self,q):
		q=q/(np.linalg.norm(q)) 
		return q

	###error calculation for line search
	def error_calc(self,alpha,q_cur,qdot_star,pd,pd_normal):
		q_next=q_cur+alpha*qdot_star
		pose_next=self.robot1.fwd(q_next)
		return np.linalg.norm(pose_next.p-pd)+np.linalg.norm(pose_next.R[:,-1]-pd_normal)

	def single_arm_stepwise_optimize(self,q_init,curve=[],curve_normal=[]):
		if len(curve)==0:
			curve=self.curve
			curve_normal=self.curve_normal
		q_all=[q_init]
		q_out=[q_init]
		Kw=1
		for i in range(len(curve)):
			print(i)
			try:
				error_fb=999
				error_fb_prev=999

				while error_fb>0.01:

					pose_now=self.robot1.fwd(q_all[-1])
					error_fb=np.linalg.norm(pose_now.p-curve[i])+Kw*np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
					
					Kq=.01*np.eye(6)    #small value to make sure positive definite
					KR=np.eye(3)        #gains for position and orientation error

					J=self.robot1.jacobian(q_all[-1])        #calculate current Jacobian
					Jp=J[3:,:]
					JR=J[:3,:]
					JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

					H=np.dot(np.transpose(Jp),Jp)+Kq+Kw*np.dot(np.transpose(JR_mod),JR_mod)
					H=(H+np.transpose(H))/2

					vd=curve[i]-pose_now.p
					ezdotd=(curve_normal[i]-pose_now.R[:,-1])

					f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),ezdotd)
					qdot=solve_qp(H,f,lb=self.robot1.lower_limit-q_all[-1]+self.lim_factor*np.ones(6),ub=self.robot1.upper_limit-q_all[-1]-self.lim_factor*np.ones(6))

					#avoid getting stuck
					if abs(error_fb-error_fb_prev)<0.0001:
						break
					error_fb_prev=error_fb

					###line search
					alpha=fminbound(self.error_calc,0,0.999999999999999999999,args=(q_all[-1],qdot,curve[i],curve_normal[i],))
					if alpha<0.01:
						break
					q_all.append(q_all[-1]+alpha*qdot)
					# print(q_all[-1])
			except:
				q_out.append(q_all[-1])
				traceback.print_exc()
				raise AssertionError
				break

			q_out.append(q_all[-1])

		q_out=np.array(q_out)[1:]
		return q_out
	def error_calc2(self,alpha,q1,qdot1,pose2_world_now,curve,curve_normal):
		q1_next=q1+alpha*qdot1
		pose1_next=self.robot1.fwd(q1_next)
		return np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_next.p-pose2_world_now.p)-curve)+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_next.R[:,-1])-curve_normal)	

	def error_calc3(self,alpha,q2,qdot2,pose1_now,curve,curve_normal):
		q2_next=q2+alpha*qdot2
		pose2_world_next=self.robot2.fwd(q2_next,self.base2_R,self.base2_p)	
		return np.linalg.norm(np.dot(pose2_world_next.R.T,pose1_now.p-pose2_world_next.p)-curve)+np.linalg.norm(np.dot(pose2_world_next.R.T,pose1_now.R[:,-1])-curve_normal)	
	def error_calc4(self,alpha,q1,q2,qdot,curve,curve_normal):
		q1_next=q1+alpha*qdot[:6]
		q2_next=q2+alpha*qdot[6:]
		pose1_next=self.robot1.fwd(q1_next)
		pose2_world_next=self.robot2.fwd(q2_next,self.base2_R,self.base2_p)	
		return np.linalg.norm(np.dot(pose2_world_next.R.T,pose1_next.p-pose2_world_next.p)-curve)+np.linalg.norm(np.dot(pose2_world_next.R.T,pose1_next.R[:,-1])-curve_normal)	

	def dual_arm_stepwise_optimize(self,q_init1,q_init2):
		#curve_normal: expressed in second robot tool frame
		###all (jacobian) in robot2 tool frame
		q_all1=[q_init1]
		q_out1=[q_init1]
		q_all2=[q_init2]
		q_out2=[q_init2]

		#####weights
		Kw=0.1
		Kq=.01*np.eye(12)    #small value to make sure positive definite
		KR=np.eye(3)        #gains for position and orientation error

		###concatenated bounds
		joint_vel_limit=np.hstack((self.robot1.joint_vel_limit,self.robot2.joint_vel_limit))
		upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
		lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))
		joint_acc_limit=np.hstack((self.robot1.joint_acc_limit,self.robot2.joint_acc_limit))

		###moving weights, p1 p2,r1 r2
		weight_distro=np.array([[1,1],
								[1,1]])

		for i in range(len(self.curve)):
			print(i)
			try:
				error_fb=999
				while error_fb>0.2:
					pose1_now=self.robot1.fwd(q_all1[-1])
					pose2_now=self.robot2.fwd(q_all2[-1])

					pose2_world_now=self.robot2.fwd(q_all2[-1],self.base2_R,self.base2_p)

					error_fb=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])	

					# print(i)
					# print(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])
					# print(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])
					########################################################QP formation###########################################
					
					J1=self.robot1.jacobian(q_all1[-1])        #calculate current Jacobian
					J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
					J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
					J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

					J2=self.robot2.jacobian(q_all2[-1])        #calculate current Jacobian, mapped to robot2 tool frame
					J2p=np.dot(pose2_now.R.T,J2[3:,:])
					J2R=np.dot(pose2_now.R.T,J2[:3,:])
					J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

					#form 6x12 jacobian with weight distribution
					J_all_p=np.hstack((weight_distro[0,0]*J1p,-weight_distro[0,1]*J2p))
					J_all_R=np.hstack((weight_distro[1,0]*J1R_mod,-weight_distro[1,1]*J2R_mod))
					
					H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
					H=(H+np.transpose(H))/2

					vd=self.curve[i]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)  
					ezdotd=self.curve_normal[i]-np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])

					f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)
					qdot=solve_qp(H,f,lb=lower_limit-np.hstack((q_all1[-1],q_all2[-1]))+self.lim_factor*np.ones(12),ub=upper_limit-np.hstack((q_all1[-1],q_all2[-1]))-self.lim_factor*np.ones(12))

					# alpha=fminbound(self.error_calc4,0,0.999999999999999999999,args=(q_all1[-1],q_all2[-1],qdot,self.curve[i],self.curve_normal[i],))
					# print(alpha)
					alpha=1
					q_all1.append(q_all1[-1]+alpha*qdot[:6])
					q_all2.append(q_all2[-1]+alpha*qdot[6:])

			except:
				traceback.print_exc()
				q_out1.append(q_all1[-1])
				q_out2.append(q_all2[-1])			
				raise AssertionError
				break

			q_out1.append(q_all1[-1])
			q_out2.append(q_all2[-1])

		q_out1=np.array(q_out1)[1:]
		q_out2=np.array(q_out2)[1:]
		return q_out1, q_out2

	def dual_arm_stepwise_optimize_separate(self,q_init1,q_init2):
		#QP motion solver for robot1 fixed position, robot2 fixed orientation
		#curve_normal: expressed in second robot tool frame
		###all (jacobian) in robot2 tool frame
		q_all1=[q_init1]
		q_out1=[q_init1]
		q_all2=[q_init2]
		q_out2=[q_init2]

		#####weights
		Kw=0.1
		Kq=.01*np.eye(6)    #small value to make sure positive definite
		KR=np.eye(3)        #gains for position and orientation error

		for i in range(len(self.curve)):
			print(i)
			try:
				error_fb=999
				error_fb_prev=999

				while error_fb>0.2:
					pose1_now=self.robot1.fwd(q_all1[-1])
					pose2_now=self.robot2.fwd(q_all2[-1])

					pose2_world_now=self.robot2.fwd(q_all2[-1],self.base2_R,self.base2_p)

					error_fb=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])	

					# print(i)
					# print(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])
					# print(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])
					########################################################QP formation###########################################
					
					J1=self.robot1.jacobian(q_all1[-1])        #calculate current Jacobian
					J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
					J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
					J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

					J2=self.robot2.jacobian(q_all2[-1])        #calculate current Jacobian, mapped to robot2 tool frame
					J2p=np.dot(pose2_now.R.T,J2[3:,:])
					J2R=np.dot(pose2_now.R.T,J2[:3,:])
					J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

					###Robot 1 QP, only for orientation

					H=np.transpose(J1R_mod)@J1R_mod+Kq
					H=(H+np.transpose(H))/2

					ezdotd=self.curve_normal[i]-np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])

					f=-np.transpose(J1R_mod)@ezdotd


					q1dot=solve_qp(H,f,A=J1p,b=np.zeros(3),lb=self.robot1.lower_limit-q_all1[-1]+self.lim_factor*np.ones(6),ub=self.robot1.upper_limit-q_all1[-1]-self.lim_factor*np.ones(6))


					###Robot 1 QP, only for position
					H=np.transpose(J2p)@J2p+Kq
					H=(H+np.transpose(H))/2

					vd=self.curve[i]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)

					#negate vd here for second arm due to relative motion
					f=-np.dot(np.transpose(J2p),-vd)
					q2dot=solve_qp(H,f,A=J2R,b=np.zeros(3),lb=self.robot2.lower_limit-q_all2[-1]+self.lim_factor*np.ones(6),ub=self.robot2.upper_limit-q_all2[-1]-self.lim_factor*np.ones(6))

					
					alpha=1
					q_all1.append(q_all1[-1]+alpha*q1dot)
					q_all2.append(q_all2[-1]+alpha*q2dot)
			except:
				traceback.print_exc()
				q_out1.append(q_all1[-1])
				q_out2.append(q_all2[-1])			
				raise AssertionError
				break

			q_out1.append(q_all1[-1])
			q_out2.append(q_all2[-1])

		q_out1=np.array(q_out1)[1:]
		q_out2=np.array(q_out2)[1:]
		return q_out1, q_out2



	def orientation_interp(self,R_init,R_end,steps):
		curve_fit_R=[]
		###find axis angle first
		R_diff=np.dot(R_init.T,R_end)
		k,theta=R2rot(R_diff)
		for i in range(steps):
			###linearly interpolate angle
			angle=theta*i/(steps-1)
			R=rot(k,angle)
			curve_fit_R.append(np.dot(R_init,R))
		curve_fit_R=np.array(curve_fit_R)
		return curve_fit_R


	def inv_all(self,curve,curve_R,q_init):
		curve_js=[q_init]
		for i in range(1,len(curve)):
			try:
				q_all=np.array(self.robot1.inv(curve[i],curve_R[i]))
			except:
				traceback.print_exc()
				pass
			###choose inv_kin closest to previous joints
			try:
				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				curve_js.append(q_all[order[0]])

			except:
				traceback.print_exc()
				return
		return curve_js
	def curve_pose_opt(self,x):
		###optimize on curve pose for single arm
		theta0=np.linalg.norm(x[:3])	###pose rotation angle
		k=x[:3]/theta0					###pose rotation axis
		shift=x[3:-1]					###pose translation
		theta1=x[-1]					###initial spray angle (1DOF)

		R_curve=rot(k,theta0)
		curve_new=np.dot(R_curve,self.curve.T).T+np.tile(shift,(len(self.curve),1))
		curve_normal_new=np.dot(R_curve,self.curve_normal.T).T

		R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
		R=np.dot(R_temp,Rz(theta1))
		try:
			q_init=self.robot1.inv(curve_new[0],R)[0]
			q_out=self.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
			
		except:
			# traceback.print_exc()
			return 999
		
		dlam=calc_lamdot(q_out,self.lam,self.robot1,1)

		
		print(min(dlam))
		return -min(dlam)

	def dual_arm_opt(self,x):
		q_init2=x[:-1]

		pose2_world_now=self.robot2.fwd(q_init2,self.base2_R,self.base2_p)

		R_temp=direction2R(np.dot(pose2_world_now.R,self.curve_normal[0]),-self.curve[1]+self.curve[0])
		R=np.dot(R_temp,Rz(x[-1]))
		try:
			q_init1=self.robot1.inv(pose2_world_now.p,R)[0]
			q_out1,q_out2=self.dual_arm_stepwise_optimize(q_init1,q_init2)
		except:
			# traceback.print_exc()
			return 999

		dlam=calc_lamdot_2arm(np.hstack((q_out1,q_out2)),self.lam,robot1,self.robot2,step=1)

		print(min(dlam))
		return -min(dlam)

	def single_arm_theta0_opt(self,theta0):

		R_temp=direction2R(self.curve_normal[0],-self.curve[1]+self.curve[0])
		R=np.dot(R_temp,Rz(theta0[0]))
		try:
			q_init=self.robot1.inv(self.curve[0],R)[0]
			q_out=self.single_arm_stepwise_optimize(q_init)
		except:
			return 999

		
		dlam=calc_lamdot(q_out,self.lam,self.robot1,1)
		print(min(dlam))
		return -min(dlam)

	def single_arm_global_opt(self,x):
		####x: [pose_choice, theta@breakpoints]
		theta=x[1:]
		pose_choice=int(np.floor(x[0]))

		for i in range(len(self.curve)):
			if i==0:

				R_temp=direction2R(self.curve_normal[0],-self.curve_original[1]+self.curve[0])


				R=np.dot(R_temp,Rz(theta[i]))
				try:
					q_out=[self.robot1.inv(self.curve[i],R)[pose_choice]]
				except:
					traceback.print_exc()
					return 999

			else:
				R_temp=direction2R(self.curve_normal[i],-self.curve[i]+self.curve_original[self.act_breakpoints[i]-1])

				R=np.dot(R_temp,Rz(theta[i]))
				try:
					###get closet config to previous one
					q_inv_all=self.robot1.inv(self.curve[i],R)
					temp_q=q_inv_all-q_out[-1]
					order=np.argsort(np.linalg.norm(temp_q,axis=1))
					q_out.append(q_inv_all[order[0]])
				except:
					# traceback.print_exc()
					return 999

		dlam=calc_lamdot(q_out,self.lam,self.robot1,1)
		print(min(dlam))
		return -min(dlam)	

	def single_arm_global_opt_blended(self,x):
		####x: [pose_choice, theta@breakpoints]
		theta=x[1:]
		pose_choice=int(np.floor(x[0]))


		for i in range(len(self.curve)):
			if i==0:

				R_temp=direction2R(self.curve_normal[0],-self.curve_original[1]+self.curve[0])


				R=np.dot(R_temp,Rz(theta[i]))
				try:
					q_out=[self.robot1.inv(self.curve[i],R)[pose_choice]]
				except:
					traceback.print_exc()
					return 999

			else:
				R_temp=direction2R(self.curve_normal[i],-self.curve[i]+self.curve_original[self.act_breakpoints[i]-1])

				R=np.dot(R_temp,Rz(theta[i]))
				try:
					###get closet config to previous one
					q_inv_all=self.robot1.inv(self.curve[i],R)
					temp_q=q_inv_all-q_out[-1]
					order=np.argsort(np.linalg.norm(temp_q,axis=1))
					q_out.append(q_inv_all[order[0]])
				except:
					# traceback.print_exc()
					return 999

		# curve_blend_js,dqdlam_list,spl_list,merged_idx=blend_js2(q_out,self.breakpoints,self.lam_original)
		# lamdot_min=est_lamdot_min(dqdlam_list,self.breakpoints,self.lam_original,spl_list,merged_idx,self.robot1)
		lam_blended,q_blended=blend_cs(q_out,self.curve_original,self.breakpoints,self.lam_original,self.primitives,self.robot1)
		dlam=calc_lamdot(q_blended,lam_blended,self.robot1,1)
		lamdot_min=min(dlam)
		print(lamdot_min)
		return -lamdot_min	


	def curve_pose_opt_blended(self,x):
		###optimize on curve pose for single arm
		####x: [pose_choice,blade k*theta, blade position, theta@breakpoints]
		pose_choice=int(np.floor(x[0]))
		blade_theta=np.linalg.norm(x[1:4])	###pose rotation angle
		k=x[1:4]/blade_theta					###pose rotation axis
		shift=x[4:7]					###pose translation
		theta=x[7:]					###remaining DOF @breakpoints

		R_curve=rot(k,blade_theta)
		curve_new=np.dot(R_curve,self.curve.T).T+np.tile(shift,(len(self.curve),1))
		curve_normal_new=np.dot(R_curve,self.curve_normal.T).T
		curve_originial_new=np.dot(R_curve,self.curve_original.T).T+np.tile(shift,(len(self.curve_original),1))


		for i in range(len(self.curve)):
			if i==0:
				R_temp=direction2R(curve_normal_new[0],-curve_originial_new[1]+curve_new[0])

				R=np.dot(R_temp,Rz(theta[i]))
				try:
					q_out=[self.robot1.inv(curve_new[i],R)[pose_choice]]
				except:
					# traceback.print_exc()
					return 999

			else:
				R_temp=direction2R(curve_normal_new[i],-curve_new[i]+curve_originial_new[self.act_breakpoints[i]-1])

				R=np.dot(R_temp,Rz(theta[i]))
				try:
					###get closet config to previous one
					q_inv_all=self.robot1.inv(curve_new[i],R)
					temp_q=q_inv_all-q_out[-1]
					order=np.argsort(np.linalg.norm(temp_q,axis=1))
					q_out.append(q_inv_all[order[0]])
				except:
					# traceback.print_exc()
					return 999

		# curve_blend_js,dqdlam_list,spl_list,merged_idx=blend_js2(q_out,self.breakpoints,self.lam_original)
		# lamdot_min=est_lamdot_min(dqdlam_list,self.breakpoints,self.lam_original,spl_list,merged_idx,self.robot1)
		# lam_blended,q_blended=blend_cs(q_out,curve_originial_new,self.breakpoints,self.lam_original,self.primitives,self.robot1)
		try:
			lam_blended,q_blended=blend_js_from_primitive(q_out,curve_originial_new,self.breakpoints,self.lam_original,self.primitives,self.robot1)
		except:
			return 999
		dlam=calc_lamdot(q_blended,lam_blended,self.robot1,1)
		lamdot_min=min(dlam)
		print(lamdot_min)
		return -lamdot_min	


	def calc_js_from_theta(self,theta,curve,curve_normal):
		###solve inv given 6th dof constraint theta
		for i in range(len(curve)):
			if i==0:
				R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
				R=np.dot(R_temp,Rz(theta[i]))
				try:
					q_out=[self.robot1.inv(curve[i],R)[0]]
				except:
					traceback.print_exc()
					return 999

			else:
				R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
				R=np.dot(R_temp,Rz(theta[i]))
				try:
					###get closet config to previous one
					q_inv_all=self.robot1inv(curve[i],R)
					temp_q=q_inv_all-q_out[-1]
					order=np.argsort(np.linalg.norm(temp_q,axis=1))
					q_out.append(q_inv_all[order[0]])
				except:
					# traceback.print_exc()
					return 999


		return calc_lamdot(q_out,self.lam,self.robot_1,1)


def main():
	return 

if __name__ == "__main__":
	main()
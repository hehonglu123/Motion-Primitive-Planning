import numpy as np
from tqdm import tqdm
from pandas import *
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, shgo, NonlinearConstraint, minimize, fminbound, Bounds
from scipy.linalg import qr
from qpsolvers import solve_qp
from lambda_calc_old import *

def reduce2full_row_rank(A, b):
	# Perform QR decomposition with pivoting
	Q, R, P = qr(A.T, pivoting=True)

	# Determine the rank and select the first 5 independent rows
	rank = np.linalg.matrix_rank(A)
	independent_rows = P[:rank]

	# Select the first 5 linearly independent rows from A and corresponding entries from b
	A_reduced = A[independent_rows[:5], :]
	b_reduced = b[independent_rows[:5]]

	return A_reduced, b_reduced

def dndspherical_Jacobian(n,JR_mod):
	# Calculate the Jacobian of the spherical coordinates
	##Given a unit vector n, compute updated reduced DoF Jacobian
	# J_spherical = np.zeros((3, 3))
	# J_spherical[0]=n
	# J_spherical[1]=np.array([n[0]*n[2],n[1]*n[2],-n[0]**2-n[1]**2])/np.sqrt(n[0]**2+n[1]**2)
	# J_spherical[2]=np.array([-n[1],n[0],0])/np.sqrt(n[0]**2+n[1]**2)

	spherical=Cartesian2Spherical(n)
	J_spherical=np.array([[np.sin(spherical[0])*np.cos(spherical[1]),np.cos(spherical[0])*np.cos(spherical[1]),-np.sin(spherical[0])*np.sin(spherical[1])],
		[np.sin(spherical[1])*np.sin(spherical[0]),np.cos(spherical[0])*np.sin(spherical[1]),np.sin(spherical[0])*np.cos(spherical[1])],
		[np.cos(spherical[0]),-np.sin(spherical[0]),0]])
	J_spherical=np.linalg.inv(J_spherical)

	J_reduced = J_spherical @ JR_mod
	return J_reduced[1:]

def Cartesian2Spherical(n):
	# Convert Cartesian coordinates to spherical coordinates
	
	theta = np.arctan2(np.sqrt(n[0]**2 + n[1]**2), n[2])
	if n[1]==0:
		phi=0
	else:
		phi = np.arctan2(n[1], n[0])

	return np.array([theta, phi])


class lambda_opt(object):
	###robot1 hold paint gun, robot2 hold part
	def __init__(self,curve,curve_normal,robot1,robot2=None,urdf_path=None,curve_name='',steps=50,breakpoints=[],primitives=[],v_cmd=1000):

		self.curve_original=curve
		self.curve_normal_original=curve_normal
		self.robot1=robot1
		self.robot2=robot2
		self.urdf_path=urdf_path
		if urdf_path:
			self.tes_env=Tess_Env(urdf_path)
			self.tes_env.update_pose(self.robot1.robot_name,np.eye(4))
		else:
			self.tes_env=None

		self.curve_name=curve_name

		self.v_cmd=v_cmd

		self.steps=steps

		self.lim_factor=0.0001###avoid fwd error on joint limit

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
			#get rid of duplicates
			self.act_breakpoints=np.unique(self.act_breakpoints)
			self.curve=curve[self.act_breakpoints]
			self.curve_normal=curve_normal[self.act_breakpoints]

		###find path length
		self.lam_original=calc_lam_cs(curve)
		self.lam=self.lam_original[self.act_breakpoints]

	def __getstate__(self):
		state = self.__dict__.copy()
		if self.urdf_path:
			del state['tes_env']
		return state

	def __setstate__(self, state):
		# Restore instance attributes (tesseract).
		self.__dict__.update(state)
		if self.urdf_path:
			self.tes_env=Tess_Env(self.urdf_path)
			self.tes_env.update_pose(self.robot1.robot_name,np.eye(4))


	def normalize_dq(self,q):
		q=q/(np.linalg.norm(q)) 
		return q

	###error calculation for line search
	def error_calc(self,alpha,q_cur,qdot_star,pd,pd_normal):
		q_next=q_cur+alpha*qdot_star
		pose_next=self.robot1.fwd(q_next)
		return np.linalg.norm(pose_next.p-pd)+np.linalg.norm(pose_next.R[:,-1]-pd_normal)
	
	def single_arm_stepwise_optimize(self,q_init,curve=[],curve_normal=[], using_spherical=False):
		if len(curve)==0:
			curve=self.curve
			curve_normal=self.curve_normal
		q_all=[q_init]
		q_out=[q_init]
		Kw=1
		for i in tqdm(range(len(curve))):
			# print(i)
			try:
				now=time.time()
				error_fb=999
				error_fb_prev=999
				error_fb_w=999

				while error_fb>0.0001 and error_fb_w>0.0000000001:
					if time.time()-now>1:
						print('qp timeout')
						raise AssertionError
						break

					pose_now=self.robot1.fwd(q_all[-1])
					error_fb=np.linalg.norm(pose_now.p-curve[i])+Kw*np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
					error_fb_w=np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
					
					Kq=.01*np.eye(6)    #small value to make sure positive definite
					KR=np.eye(3)        #gains for position and orientation error

					vd=curve[i]-pose_now.p
					J=self.robot1.jacobian(q_all[-1])        #calculate current Jacobian
					Jp=J[3:,:]
					JR=J[:3,:]
					JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

					
					if using_spherical:
						###using spherical coordinates
						JR_mod=dndspherical_Jacobian(pose_now.R[:,-1],JR_mod)
						dspherical_des=Cartesian2Spherical(curve_normal[i])-Cartesian2Spherical(pose_now.R[:,-1])
						f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),dspherical_des)
					else:
						###using ezcross
						ezdotd=(curve_normal[i]-pose_now.R[:,-1])
						f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),ezdotd)

					H=np.dot(np.transpose(Jp),Jp)+Kq+Kw*np.dot(np.transpose(JR_mod),JR_mod)
					H=(H+np.transpose(H))/2
					qdot=solve_qp(H,f,lb=(self.robot1.lower_limit+0.1)-q_all[-1]+self.lim_factor*np.ones(6),ub=(self.robot1.upper_limit-0.1)-q_all[-1]-self.lim_factor*np.ones(6),solver='cvxopt')
					
					#avoid getting stuck
					if abs(error_fb-error_fb_prev)<0.0001:
						raise AssertionError("Stuck")

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

	def single_arm_stepwise_optimize2(self,q_init,lamdot_des,curve=[],curve_normal=[],using_spherical=False):
		### redundancy resolution with qdot and qddot constraint, with equality constraint Jq=nu 7/21/2024
		# q_init: initial joint position
		# lamdot_des: desired lambdadot (\mu)

		if len(curve)==0:
			curve=self.curve
			curve_normal=self.curve_normal
		lam=self.lam

		q_all=[q_init]
		
		qdot_all=[]

		for i in tqdm(range(len(curve)-1)):

			dt=(lam[i+1]-lam[i])/lamdot_des
			pose_now=self.robot1.fwd(q_all[-1])
			

			J=self.robot1.jacobian(q_all[-1])        #calculate current Jacobian
			Jp=J[3:,:]
			JR=J[:3,:]
			JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)
			
			
			Ka=0.1*np.eye(6)    #qddot opt weights
			vd=(curve[i+1]-curve[i])/dt

			J_mod=np.vstack((Jp,JR_mod))

			
			if using_spherical:
			###using spherical coordinates
				JR_mod=dndspherical_Jacobian(pose_now.R[:,-1],JR_mod)
				dspherical_des=(Cartesian2Spherical(curve_normal[i+1])-Cartesian2Spherical(curve_normal[i]))/dt
				A = np.vstack((Jp,JR_mod))
				b = np.hstack((vd,dspherical_des))
			###using ezcross
			else:
				ezdotd=(curve_normal[i+1]-curve_normal[i])/dt
				A, b = reduce2full_row_rank(J_mod, np.hstack((vd,ezdotd)))

			if len(qdot_all)<2:
				H=np.eye(6)
				lb = np.maximum(-self.robot1.joint_vel_limit, (self.robot1.lower_limit + self.lim_factor * np.ones(6) - q_all[-1])/dt)
				ub = np.minimum(self.robot1.joint_vel_limit, (self.robot1.upper_limit - self.lim_factor * np.ones(6) - q_all[-1])/dt)
				f=np.zeros(6)

			else:
				H=np.eye(6)+Ka
				f=-2*Ka@qdot_all[-1]
				qddot_lim=self.robot1.get_acc(q_all[-1],direction=np.array((1+np.sign(qdot_all[-1]-qdot_all[-2]))/2).astype(int)[:3])
				lb = np.maximum(np.maximum(-self.robot1.joint_vel_limit, (self.robot1.lower_limit + self.lim_factor * np.ones(6) - q_all[-1])/dt), qdot_all[-1] - qddot_lim * dt)
				ub = np.minimum(np.minimum(self.robot1.joint_vel_limit, (self.robot1.upper_limit - self.lim_factor * np.ones(6) - q_all[-1])/dt), qdot_all[-1] + qddot_lim * dt)
			
			qdot=solve_qp(H,f,A=A,b=b,lb=lb,ub=ub,solver='cvxopt')
			
			if qdot is None:
				raise AssertionError('no solution available')
			
			q_seed=q_all[-1]+qdot*dt

			###Iterative till exact solution
			try:
				now=time.time()
				error_fb=999
				error_fb_prev=999

				while error_fb>0.01:
					if time.time()-now>1:
						print('qp timeout')
						raise AssertionError
					
					# print(np.all(q_seed>self.robot1.lower_limit) & np.all(q_seed<self.robot1.upper_limit))
					pose_now=self.robot1.fwd(q_seed)
					error_fb=np.linalg.norm(pose_now.p-curve[i])+np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
					
					Kq=.01*np.eye(6)    #small value to make sure positive definite

					J=self.robot1.jacobian(q_seed)        #calculate current Jacobian
					Jp=J[3:,:]
					JR=J[:3,:]
					JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

					vd=curve[i]-pose_now.p

					if using_spherical:
						###using spherical coordinates
						JR_mod=dndspherical_Jacobian(pose_now.R[:,-1],JR_mod)
						dspherical_des=Cartesian2Spherical(curve_normal[i])-Cartesian2Spherical(pose_now.R[:,-1])
						f=-np.dot(np.transpose(Jp),vd)-np.dot(np.transpose(JR_mod),dspherical_des)
					
					else:
						###using ezcross
						ezdotd=(curve_normal[i]-pose_now.R[:,-1])
						f=-np.dot(np.transpose(Jp),vd)-np.dot(np.transpose(JR_mod),ezdotd)



					H=np.dot(np.transpose(Jp),Jp)+Kq+np.dot(np.transpose(JR_mod),JR_mod)
					H=(H+np.transpose(H))/2

					qdot=solve_qp(H,f,lb=self.robot1.lower_limit+self.lim_factor*np.ones(6)-q_seed,ub=self.robot1.upper_limit-self.lim_factor*np.ones(6)-q_seed,solver='cvxopt')

					#avoid getting stuck
					if abs(error_fb-error_fb_prev)<0.0001:
						break
					error_fb_prev=error_fb

					###line search
					# alpha=fminbound(self.error_calc,0,0.999999999999999999999,args=(q_seed,qdot,curve[i],curve_normal[i],))
					# if alpha<0.01:
					# 	break

					q_seed+=qdot
			except:
				traceback.print_exc()
				raise AssertionError

			qdot_all.append((q_seed-q_all[-1])/dt)
			q_all.append(q_seed)

		q_all=np.array(q_all)


		return q_all

	def single_arm_stepwise_optimize3(self,q_init,lamdot_des,curve=[],curve_normal=[]):
		### redundancy resolution with qdot and qddot constraint, with optimization |Jq-nu| 7/23/2024
		# q_init: initial joint position
		# lamdot_des: desired lambdadot (\mu)


		if len(curve)==0:
			curve=self.curve
			curve_normal=self.curve_normal
		lam=self.lam

		q_all=[q_init]
		
		qdot_all=[]

		for i in tqdm(range(len(curve)-1)):

			dt=(lam[i+1]-lam[i])/lamdot_des
			pose_now=self.robot1.fwd(q_all[-1])
			

			J=self.robot1.jacobian(q_all[-1])        #calculate current Jacobian
			Jp=J[3:,:]
			JR=J[:3,:]
			JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)
			
			
			vd=(curve[i+1]-curve[i])/dt
			ezdotd=(curve_normal[i+1]-curve_normal[i])/dt
			nud=np.hstack((vd,ezdotd))

			J_mod=np.vstack((Jp,JR_mod))
			H = J_mod.T @ J_mod
			f = -J_mod.T @ nud


			if len(qdot_all)<2:
				lb = np.maximum(-self.robot1.joint_vel_limit, (self.robot1.lower_limit + self.lim_factor * np.ones(6) - q_all[-1])/dt)
				ub = np.minimum(self.robot1.joint_vel_limit, (self.robot1.upper_limit - self.lim_factor * np.ones(6) - q_all[-1])/dt)

			else:
				qddot_lim=self.robot1.get_acc(q_all[-1],direction=np.array((1+np.sign(qdot_all[-1]-qdot_all[-2]))/2).astype(int)[:3])
				lb = np.maximum(np.maximum(-self.robot1.joint_vel_limit, (self.robot1.lower_limit + self.lim_factor * np.ones(6) - q_all[-1])/dt), qdot_all[-1] - qddot_lim * dt)
				ub = np.minimum(np.minimum(self.robot1.joint_vel_limit, (self.robot1.upper_limit - self.lim_factor * np.ones(6) - q_all[-1])/dt), qdot_all[-1] + qddot_lim * dt)
			
			qdot=solve_qp(H,f,lb=lb,ub=ub,solver='cvxopt')
			
			if qdot is None:
				raise AssertionError('no solution available')
			
			q_seed=q_all[-1]+qdot*dt

			###Iterative till exact solution
			try:
				now=time.time()
				error_fb=999
				error_fb_prev=999
				Kq=.01*np.eye(6)    #small value to make sure positive definite

				while error_fb>0.01:
					if time.time()-now>1:
						print('qp timeout')
						raise AssertionError
					
					# print(np.all(q_seed>self.robot1.lower_limit) & np.all(q_seed<self.robot1.upper_limit))
					pose_now=self.robot1.fwd(q_seed)
					error_fb=np.linalg.norm(pose_now.p-curve[i+1])+np.linalg.norm(pose_now.R[:,-1]-curve_normal[i+1])
					

					J=self.robot1.jacobian(q_seed)        #calculate current Jacobian
					Jp=J[3:,:]
					JR=J[:3,:]
					JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

					vd=curve[i+1]-pose_now.p
					ezdotd=(curve_normal[i+1]-pose_now.R[:,-1])
					

					nud=np.hstack((vd,ezdotd))
					J_mod=np.vstack((Jp,JR_mod))
					H = J_mod.T @ J_mod + Kq
					f = -J_mod.T @ nud

					qdot=solve_qp(H,f,lb=self.robot1.lower_limit+self.lim_factor*np.ones(6)-q_seed,ub=self.robot1.upper_limit-self.lim_factor*np.ones(6)-q_seed,solver='cvxopt')

					#avoid getting stuck
					if abs(error_fb-error_fb_prev)<0.0001:
						break
					error_fb_prev=error_fb

					q_seed+=qdot
			except:
				traceback.print_exc()
				raise AssertionError

			qdot_all.append((q_seed-q_all[-1])/dt)
			q_all.append(q_seed)

		q_all=np.array(q_all)


		return q_all
	
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

	def followx(self,curve,curve_direction):
		curve_R=[]

		for i in range(len(curve)-1):

			R_curve=direction2R(curve_direction[i],-curve[i+1]+curve[i])
			curve_R.append(R_curve)

		###insert initial orientation
		curve_R.insert(0,curve_R[0])

		try:
			q_inits=np.array(self.robot1.inv(curve[0],curve_R[0]))
		except:
			raise AssertionError('no solution available')
			return []

		for q_init in q_inits:
			curve_js=np.zeros((len(curve),6))
			curve_js[0]=q_init
			for i in range(1,len(curve)):
				try:
					q_all=np.array(self.robot1.inv(curve[i],curve_R[i]))
				except:
					#if no solution
					raise AssertionError('no solution available')
					return

				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				if np.linalg.norm(q_all[order[0]]-curve_js[i-1])>0.5:
					# print('large change')
					break	#if large changes in q
				else:
					curve_js[i]=q_all[order[0]]

			#check if all q found
			if np.linalg.norm(curve_js[-1])>0:
				break
		return curve_js
	def dual_arm_stepwise_optimize(self,q_init1,q_init2,base2_R,base2_p,w1=0.01,w2=0.01,using_spherical=False):

		###w1: weight for first robot
		###w2: weight for second robot (larger weight path shorter)
		#curve_normal: expressed in second robot tool frame
		###all (jacobian) in robot2 tool frame
		q_all1=[q_init1]
		q_out1=[q_init1]
		q_all2=[q_init2]
		q_out2=[q_init2]
		j_all1=[self.robot1.jacobian(q_all1[-1])]
		j_all2=[self.robot2.jacobian(q_all2[-1])]
		j_out1=[j_all1[0]]
		j_out2=[j_all1[0]]

		#####weights
		Kw=0.1
		Kq=w1*np.eye(12)    #small value to make sure positive definite
		Kq[6:,6:]=w2*np.eye(6)		#larger weights for second robot for it moves slower
		KR=np.eye(3)        #gains for position and orientation error

		###concatenated bounds
		joint_vel_limit=np.hstack((self.robot1.joint_vel_limit,self.robot2.joint_vel_limit))
		upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
		lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))
		joint_acc_limit=np.hstack((self.robot1.joint_acc_limit,self.robot2.joint_acc_limit))

		for i in range(len(self.curve)):
			try:
				now=time.time()
				error_fb=999
				while error_fb>0.1:
					###timeout guard
					if time.time()-now>1:
						print('qp timeout')
						raise AssertionError
						break

					pose1_now=self.robot1.fwd(q_all1[-1])
					pose2_now=self.robot2.fwd(q_all2[-1])

					self.robot2.base_H=H_from_RT(base2_R,base2_p)
					pose2_world_now=self.robot2.fwd(q_all2[-1],world=True)

					error_fb=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])	

					# print(i)
					# print(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])
					# print(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])
					########################################################QP formation###########################################
					
					J1=j_all1[-1]        #current Jacobian
					J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
					J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
					J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

					J2=j_all2[-1]        #current Jacobian        #calculate current Jacobian, mapped to robot2 tool frame
					J2p=np.dot(pose2_now.R.T,J2[3:,:])
					J2R=np.dot(pose2_now.R.T,J2[:3,:])
					J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

					p12_2=pose2_world_now.R.T@(pose1_now.p-pose2_world_now.p)
					
					#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
					J_all_p=np.hstack((J1p,-J2p+hat(p12_2)@J2R))
					J_all_R=np.hstack((J1R_mod,-J2R_mod))
					
					H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
					H=(H+np.transpose(H))/2

					vd=self.curve[i]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)

					if using_spherical:
						###using spherical coordinates
						J_all_R=dndspherical_Jacobian(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1]),J_all_R)
						dspherical_des=(Cartesian2Spherical(self.curve_normal[i])-Cartesian2Spherical(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])))
						f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),dspherical_des)
					else:
						ezdotd=self.curve_normal[i]-np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
						f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)
					
					# print("SPHERICAL: ",i,Cartesian2Spherical(self.curve_normal[i]),Cartesian2Spherical(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])))
					# print("NORMAL: ",i,self.curve_normal[i],np.dot(pose2_world_now.R.T,pose1_now.R[:,-1]))
					# print("SPHERICAL ERROR: ",np.linalg.norm(dspherical_des))
					# print("ORIENTATION ERROR: ",np.linalg.norm(self.curve_normal[i]-np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])))

					qdot=solve_qp(H,f,lb=lower_limit+self.lim_factor*np.ones(12)-np.hstack((q_all1[-1],q_all2[-1])),ub=upper_limit-self.lim_factor*np.ones(12)-np.hstack((q_all1[-1],q_all2[-1])),solver='cvxopt')

					# alpha=fminbound(self.error_calc4,0,0.999999999999999999999,args=(q_all1[-1],q_all2[-1],qdot,self.curve[i],self.curve_normal[i],))
					# print(alpha)
					alpha=1
					q_all1.append(q_all1[-1]+alpha*qdot[:6])
					q_all2.append(q_all2[-1]+alpha*qdot[6:])
					j_all1.append(self.robot1.jacobian(q_all1[-1]))
					j_all2.append(self.robot2.jacobian(q_all2[-1]))

			except:
				traceback.print_exc()
				q_out1.append(q_all1[-1])
				q_out2.append(q_all2[-1])
				j_out1.append(j_all1[-1])
				j_out2.append(j_all2[-1])		
				raise AssertionError
				break

			q_out1.append(q_all1[-1])
			q_out2.append(q_all2[-1])
			j_out1.append(j_all1[-1])
			j_out2.append(j_all2[-1])

		q_out1=np.array(q_out1)[1:]
		q_out2=np.array(q_out2)[1:]
		j_out1=j_out1[1:]
		j_out2=j_out2[1:]
		return q_out1, q_out2, j_out1, j_out2

	def dual_arm_stepwise_optimize_separate(self,q_init1,q_init2,base2_R,base2_p):
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

					pose2_world_now=self.robot2.fwd(q_all2[-1],base2_R,base2_p)

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


					q1dot=solve_qp(H,f,A=J1p,b=np.zeros(3),lb=self.robot1.lower_limit-q_all1[-1]+self.lim_factor*np.ones(6),ub=self.robot1.upper_limit-q_all1[-1]-self.lim_factor*np.ones(6),solver='cvxopt')


					###Robot 1 QP, only for position
					H=np.transpose(J2p)@J2p+Kq
					H=(H+np.transpose(H))/2

					vd=self.curve[i]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)

					#negate vd here for second arm due to relative motion
					f=-np.dot(np.transpose(J2p),-vd)
					q2dot=solve_qp(H,f,A=J2R,b=np.zeros(3),lb=self.robot2.lower_limit-q_all2[-1]+self.lim_factor*np.ones(6),ub=self.robot2.upper_limit-q_all2[-1]-self.lim_factor*np.ones(6),solver='cvxopt')

					
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

	def dual_arm_stepwise_optimize2(self,q_init1,q_init2,base2_R,base2_p,lamdot_des,w1=0.01,w2=0.01,using_spherical=False):
		### redundancy resolution with qdot and qddot constraint, 7/21/2024
		# q_init: initial joint position
		# lamdot_des: desired lambdadot (\mu)

		###w1: weight for first robot
		###w2: weight for second robot (larger weight path shorter)
		#curve_normal: expressed in second robot tool frame
		###all (jacobian) in robot2 tool frame

		lam=self.lam

		q_all1=[q_init1]
		q_all2=[q_init2]
		j_all1=[self.robot1.jacobian(q_all1[-1])]
		j_all2=[self.robot2.jacobian(q_all2[-1])]
		qdot_all1=[]
		qdot_all2=[]

		#####weights
		Kw=0.1
		Kq=w1*np.eye(12)    #small value to make sure positive definite
		Kq[6:,6:]=w2*np.eye(6)		#larger weights for second robot for it moves slower

		###concatenated bounds
		joint_vel_limit=np.hstack((self.robot1.joint_vel_limit,self.robot2.joint_vel_limit))
		upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
		lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))

		for i in tqdm(range(len(self.curve)-1)):

			dt=(lam[i+1]-lam[i])/lamdot_des

			pose1_now=self.robot1.fwd(q_all1[-1])
			pose2_now=self.robot2.fwd(q_all2[-1])

			self.robot2.base_H=H_from_RT(base2_R,base2_p)
			pose2_world_now=self.robot2.fwd(q_all2[-1],world=True)

			########################################################QP formation###########################################
			
			J1=j_all1[-1]        #current Jacobian
			J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
			J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
			J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

			J2=j_all2[-1]        #current Jacobian        #calculate current Jacobian, mapped to robot2 tool frame
			J2p=np.dot(pose2_now.R.T,J2[3:,:])
			J2R=np.dot(pose2_now.R.T,J2[:3,:])
			J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

			p12_2=pose2_world_now.R.T@(pose1_now.p-pose2_world_now.p)
			
			#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
			J_all_p=np.hstack((J1p,-J2p+hat(p12_2)@J2R))
			J_all_R=np.hstack((J1R_mod,-J2R_mod))

			J_all_mod=np.vstack((J_all_p,J_all_R))
			
			Ka=0.1*np.eye(12)    #qddot opt weights
			
			vd=(self.curve[i+1]-self.curve[i])/dt

			if using_spherical:
				###using spherical coordinates
				JR_mod=dndspherical_Jacobian(self.curve_normal[i],J_all_R)
				dspherical_des=(Cartesian2Spherical(self.curve_normal[i+1])-Cartesian2Spherical(self.curve_normal[i]))/dt
				A = np.vstack((J_all_p,JR_mod))
				b = np.hstack((vd,dspherical_des))

			else:
				ezdotd=(self.curve_normal[i+1]-self.curve_normal[i])/dt
				b=np.hstack((vd,ezdotd))
				# augmented_matrix = np.hstack((J_all_mod, b.reshape(-1,1)))
				# print('RANK COMPRIONSON: ', np.linalg.matrix_rank(J_all_mod),np.linalg.matrix_rank(augmented_matrix))
				A, b = reduce2full_row_rank(J_all_mod, np.hstack((vd,ezdotd)))

			if len(qdot_all1)<2:
				H=np.eye(12)
				f=np.zeros(12)
				
				lb = np.maximum(-joint_vel_limit, (lower_limit + self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt)
				ub = np.minimum(joint_vel_limit, (upper_limit - self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt)	
				
			else:
				H=np.eye(12)+Ka
				f=-Ka@np.hstack((qdot_all1[-1],qdot_all2[-1]))
				qddot_lim1=self.robot1.get_acc(q_all1[-1],direction=np.array((1+np.sign(qdot_all1[-1]-qdot_all1[-2]))/2).astype(int)[:3])
				qddot_lim2=self.robot2.get_acc(q_all2[-1],direction=np.array((1+np.sign(qdot_all2[-1]-qdot_all2[-2]))/2).astype(int)[:3])
				qddot_lim=np.hstack((qddot_lim1,qddot_lim2))
				lb = np.maximum(np.maximum(-joint_vel_limit, (lower_limit + self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt), np.hstack((qdot_all1[-1],qdot_all2[-1])) - qddot_lim * dt)
				ub = np.minimum(np.minimum(joint_vel_limit, (upper_limit - self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt), np.hstack((qdot_all1[-1],qdot_all2[-1])) + qddot_lim * dt)
			

			qdot=solve_qp(H,f,A=A,b=b,solver='cvxopt',lb=lb,ub=ub)

			if qdot is None:
				raise AssertionError('no solution available')

			q_seed1=q_all1[-1]+qdot[:6]*dt
			q_seed2=q_all2[-1]+qdot[6:]*dt

			try:
				now=time.time()
				error_fb=999
				while error_fb>0.01:
					###timeout guard
					if time.time()-now>1:
						print('qp timeout')
						raise AssertionError

					pose1_now=self.robot1.fwd(q_seed1)
					pose2_now=self.robot2.fwd(q_seed2)

					self.robot2.base_H=H_from_RT(base2_R,base2_p)
					pose2_world_now=self.robot2.fwd(q_seed2,world=True)

					error_fb=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i+1])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i+1])	

					########################################################QP formation###########################################
					
					J1=self.robot1.jacobian(q_seed1)
					J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
					J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
					J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

					J2=self.robot2.jacobian(q_seed2)        #current Jacobian        #calculate current Jacobian, mapped to robot2 tool frame
					J2p=np.dot(pose2_now.R.T,J2[3:,:])
					J2R=np.dot(pose2_now.R.T,J2[:3,:])
					J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

					p12_2=pose2_world_now.R.T@(pose1_now.p-pose2_world_now.p)
					
					#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
					J_all_p=np.hstack((J1p,-J2p+hat(p12_2)@J2R))
					J_all_R=np.hstack((J1R_mod,-J2R_mod))
					
					H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
					H=(H+np.transpose(H))/2

					vd=self.curve[i+1]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)

					if using_spherical:
						###using spherical coordinates
						J_all_R=dndspherical_Jacobian(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1]),J_all_R)
						dspherical_des=(Cartesian2Spherical(self.curve_normal[i+1])-Cartesian2Spherical(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])))
						f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),dspherical_des)
					else:
						ezdotd=self.curve_normal[i+1]-np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
						f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)

					qdot=solve_qp(H,f,lb=lower_limit-np.hstack((q_all1[-1],q_all2[-1]))+self.lim_factor*np.ones(12),ub=upper_limit-np.hstack((q_all1[-1],q_all2[-1]))-self.lim_factor*np.ones(12),solver='cvxopt')

					q_seed1+=qdot[:6]
					q_seed2+=qdot[6:]

			except:
				traceback.print_exc()	
				raise AssertionError
		
			qdot_all1.append((q_seed1-q_all1[-1])/dt)
			qdot_all2.append((q_seed2-q_all2[-1])/dt)
			q_all1.append(q_seed1)
			q_all2.append(q_seed2)
			j_all1.append(self.robot1.jacobian(q_seed1))
			j_all2.append(self.robot2.jacobian(q_seed2))


		return np.array(q_all1), np.array(q_all2), np.array(j_all1), np.array(j_all2)



	def dual_arm_stepwise_optimize3(self,q_init1,q_init2,base2_R,base2_p,lamdot_des,w1=0.01,w2=0.01):
		### redundancy resolution with qdot and qddot constraint, min ||Jq-\nu|| 7/23/2024
		# q_init: initial joint position
		# lamdot_des: desired lambdadot (\mu)

		###w1: weight for first robot
		###w2: weight for second robot (larger weight path shorter)
		#curve_normal: expressed in second robot tool frame
		###all (jacobian) in robot2 tool frame

		lam=self.lam

		q_all1=[q_init1]
		q_all2=[q_init2]
		j_all1=[self.robot1.jacobian(q_all1[-1])]
		j_all2=[self.robot2.jacobian(q_all2[-1])]
		qdot_all1=[]
		qdot_all2=[]

		#####weights
		Kw=0.1
		Kq=w1*np.eye(12)    #small value to make sure positive definite
		Kq[6:,6:]=w2*np.eye(6)		#larger weights for second robot for it moves slower

		###concatenated bounds
		joint_vel_limit=np.hstack((self.robot1.joint_vel_limit,self.robot2.joint_vel_limit))
		upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
		lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))

		for i in tqdm(range(len(self.curve)-1)):

			dt=(lam[i+1]-lam[i])/lamdot_des

			pose1_now=self.robot1.fwd(q_all1[-1])
			pose2_now=self.robot2.fwd(q_all2[-1])

			self.robot2.base_H=H_from_RT(base2_R,base2_p)
			pose2_world_now=self.robot2.fwd(q_all2[-1],world=True)

			########################################################QP formation###########################################
			
			J1=j_all1[-1]        #current Jacobian
			J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
			J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
			J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

			J2=j_all2[-1]        #current Jacobian        #calculate current Jacobian, mapped to robot2 tool frame
			J2p=np.dot(pose2_now.R.T,J2[3:,:])
			J2R=np.dot(pose2_now.R.T,J2[:3,:])
			J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

			p12_2=pose2_world_now.R.T@(pose1_now.p-pose2_world_now.p)
			
			#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
			J_all_p=np.hstack((J1p,-J2p+hat(p12_2)@J2R))
			J_all_R=np.hstack((J1R_mod,-J2R_mod))

			vd=(self.curve[i+1]-self.curve[i])/dt
			ezdotd=(self.curve_normal[i+1]-self.curve_normal[i])/dt

			# ###pos, ori separated
			# H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
			# f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)

			###stacked together
			J_all_mod=np.vstack((J_all_p,J_all_R))
			nud=np.hstack((vd,ezdotd))
			H=J_all_mod.T@J_all_mod
			f=-J_all_mod.T@nud

			if len(qdot_all1)<2:
				lb = np.maximum(-joint_vel_limit, (lower_limit + self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt)
				ub = np.minimum(joint_vel_limit, (upper_limit - self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt)	
				
			else:
				qddot_lim1=self.robot1.get_acc(q_all1[-1],direction=np.array((1+np.sign(qdot_all1[-1]-qdot_all1[-2]))/2).astype(int)[:3])
				qddot_lim2=self.robot2.get_acc(q_all2[-1],direction=np.array((1+np.sign(qdot_all2[-1]-qdot_all2[-2]))/2).astype(int)[:3])
				qddot_lim=np.hstack((qddot_lim1,qddot_lim2))
				lb = np.maximum(np.maximum(-joint_vel_limit, (lower_limit + self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt), np.hstack((qdot_all1[-1],qdot_all2[-1])) - qddot_lim * dt)
				ub = np.minimum(np.minimum(joint_vel_limit, (upper_limit - self.lim_factor * np.ones(12) - np.hstack((q_all1[-1],q_all2[-1])))/dt), np.hstack((qdot_all1[-1],qdot_all2[-1])) + qddot_lim * dt)
			

			qdot=solve_qp(H,f,solver='cvxopt',lb=lb,ub=ub)

			if qdot is None:
				raise AssertionError('no solution available')

			q_seed1=q_all1[-1]+qdot[:6]*dt
			q_seed2=q_all2[-1]+qdot[6:]*dt

			try:
				now=time.time()
				error_fb=999
				while error_fb>0.01:
					###timeout guard
					if time.time()-now>1:
						print('qp timeout')
						raise AssertionError

					pose1_now=self.robot1.fwd(q_seed1)
					pose2_now=self.robot2.fwd(q_seed2)

					self.robot2.base_H=H_from_RT(base2_R,base2_p)
					pose2_world_now=self.robot2.fwd(q_seed2,world=True)

					error_fb=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i+1])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i+1])	

					########################################################QP formation###########################################
					
					J1=self.robot1.jacobian(q_seed1)
					J1p=np.dot(pose2_world_now.R.T,J1[3:,:])
					J1R=np.dot(pose2_world_now.R.T,J1[:3,:])
					J1R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J1R)

					J2=self.robot2.jacobian(q_seed2)        #current Jacobian        #calculate current Jacobian, mapped to robot2 tool frame
					J2p=np.dot(pose2_now.R.T,J2[3:,:])
					J2R=np.dot(pose2_now.R.T,J2[:3,:])
					J2R_mod=-np.dot(hat(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])),J2R)

					p12_2=pose2_world_now.R.T@(pose1_now.p-pose2_world_now.p)
					
					#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
					J_all_p=np.hstack((J1p,-J2p+hat(p12_2)@J2R))
					J_all_R=np.hstack((J1R_mod,-J2R_mod))
					

					vd=self.curve[i+1]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)
					ezdotd=self.curve_normal[i+1]-np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
					

					###pos, ori separated
					# H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
					# f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)


					###stacked together
					nud=np.hstack((vd,ezdotd))
					J_all_mod=np.vstack((J_all_p,J_all_R))
					H=J_all_mod.T@J_all_mod + Kq
					f=-J_all_mod.T@nud

					qdot=solve_qp(H,f,lb=lower_limit-np.hstack((q_all1[-1],q_all2[-1]))+self.lim_factor*np.ones(12),ub=upper_limit-np.hstack((q_all1[-1],q_all2[-1]))-self.lim_factor*np.ones(12),solver='cvxopt')

					q_seed1+=qdot[:6]
					q_seed2+=qdot[6:]

			except:
				traceback.print_exc()	
				raise AssertionError
		
			qdot_all1.append((q_seed1-q_all1[-1])/dt)
			qdot_all2.append((q_seed2-q_all2[-1])/dt)
			q_all1.append(q_seed1)
			q_all2.append(q_seed2)
			j_all1.append(self.robot1.jacobian(q_seed1))
			j_all2.append(self.robot2.jacobian(q_seed2))


		return np.array(q_all1), np.array(q_all2), np.array(j_all1), np.array(j_all2)
	
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
	def curve_pose_opt(self,x,method=1):
		###optimize on curve pose for single arm with lambda dot calculation
		theta0=np.linalg.norm(x[:3])	###pose rotation angle
		k=x[:3]/theta0					###pose rotation axis
		shift=x[3:-1]					###pose translation
		theta1=x[-1]					###initial spray angle (1DOF)

		R_curve=rot(k,theta0)
		curve_new=np.dot(R_curve,self.curve.T).T+np.tile(shift,(len(self.curve),1))
		curve_normal_new=np.dot(R_curve,self.curve_normal.T).T

		###make sure curve above ground:
		if min(curve_new[:,3])<0:
			return 999

		R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
		R=np.dot(R_temp,Rz(theta1))
		try:
			q_init=self.robot1.inv(curve_new[0],R)[0]
			if method==1:###follow +x
				q_out=self.followx(curve_new,curve_normal_new)
			else:
				q_out=self.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
			
		except:
			# traceback.print_exc()
			return 999

		###make sure extension possible by checking start & end configuration
		if np.min(self.robot1.upper_limit-q_out[0])<0.2 or  np.min(q_out[0]-self.robot1.lower_limit)<0.2 or np.min(self.robot1.upper_limit-q_out[-1])<0.2 or  np.min(q_out[-1]-self.robot1.lower_limit)<0.2:
			return 999
		
		dlam=calc_lamdot(q_out,self.lam,self.robot1,1)

		
		print(min(dlam))
		return -min(dlam)

	def curve_pose_opt2(self,x,method=2):
		###optimize on curve pose for single arm with speed estimation
		theta0=np.linalg.norm(x[:3])	###pose rotation angle
		k=x[:3]/theta0					###pose rotation axis
		shift=x[3:-1]					###pose translation
		theta1=x[-1]					###initial spray angle (1DOF)

		R_curve=rot(k,theta0)
		curve_new=np.dot(R_curve,self.curve.T).T+np.tile(shift,(len(self.curve),1))
		curve_normal_new=np.dot(R_curve,self.curve_normal.T).T

		###make sure curve above ground:
		if np.min(curve_new[:,2])<0:
			return 999

		R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
		R=np.dot(R_temp,Rz(theta1))
		try:
			q_init=self.robot1.inv(curve_new[0],R)[0]
			if method==1:###follow +x
				q_out=self.followx(curve_new,curve_normal_new)
			else:
				q_out=self.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
			
		except:
			# traceback.print_exc()
			return 999
		
		###make sure extension possible by checking start & end configuration
		ext_threshold=0.3
		if np.min(self.robot1.upper_limit-q_out[0])<ext_threshold or  np.min(q_out[0]-self.robot1.lower_limit)<ext_threshold or np.min(self.robot1.upper_limit-q_out[-1])<ext_threshold or  np.min(q_out[-1]-self.robot1.lower_limit)<ext_threshold:
			return 999

		speed=traj_speed_est(self.robot1,q_out,self.lam,self.v_cmd)

		###TODO: FIX TESSERACT MEMORY ERROR
		if self.tes_env:
			self.tes_env.update_pose(self.curve_name,H_from_RT(R_curve,shift/1000.))
			if self.tes_env.check_collision_single(self.robot1.robot_name,self.curve_name,q_out):
				return 999

		return -min(speed)

	def single_arm_constraint(self,q_flat):
		###Formulate Equality Constraint (Tracking IK) for single arm
		try:
			q_all=np.reshape(q_flat,(len(q_flat)//6,6))
			error_all=[]
			for i in range(len(q_all)):
				pose_now=self.robot1.fwd(q_all[i])
				error_p=np.linalg.norm(pose_now.p-self.curve[i])
				error_w=np.linalg.norm(pose_now.R[:,-1]-self.curve_normal[i])
				
				error_all.append(error_p+error_w)
		except:
			# traceback.print_exc()
			# raise AssertionError
			return 999
		
		return np.mean(error_all)

	def single_arm_objective(self,q_flat):
		### Formulate Objective Function for single arm, minimum lamdot
		try:
			lamdot_boundary=lambdadot_qlambda(self.robot1,np.reshape(q_flat,(len(q_flat)//6,6)),self.lam)
		except:
			traceback.print_exc()
			raise AssertionError
		
		return -np.min(lamdot_boundary)


	def dual_arm_init_opt(self,x):
		q_init2=x[:-1]

		pose2_world_now=self.robot2.fwd(q_init2,world=True)

		R_temp=direction2R(pose2_world_now.R@self.curve_normal[0],-self.curve[1]+self.curve[0])
		R=np.dot(R_temp,Rz(x[-1]))
		try:
			q_init1=self.robot1.inv(pose2_world_now.p,R)[0]
			q_out1,q_out2=self.dual_arm_stepwise_optimize(q_init1,q_init2,w1=0.02,w2=0.01)
		except:
			# traceback.print_exc()
			return 999

		###make sure extension possible by checking start & end configuration
		if np.min(self.robot1.upper_limit-q_out1[0])<0.2 or  np.min(q_out1[0]-self.robot1.lower_limit)<0.2 or np.min(self.robot1.upper_limit-q_out1[-1])<0.2 or np.min(q_out1[-1]-self.robot1.lower_limit)<0.2:
			return 999

		###make sure extension possible by checking start & end configuration
		if np.min(self.robot2.upper_limit-q_out2[0])<0.2 or  np.min(q_out2[0]-self.robot2.lower_limit)<0.2 or np.min(self.robot2.upper_limit-q_out2[-1])<0.2 or  np.min(q_out2[-1]-self.robot2.lower_limit)<0.2:
			return 999

		# dlam=calc_lamdot_2arm(np.hstack((q_out1,q_out2)),self.lam,self.robot1,self.robot2,step=1)
		speed,_,_=traj_speed_est_dual(self.robot1,self.robot2,q_out1,q_out2,self.lam,self.v_cmd)

		return -min(speed)

	def dual_arm_opt_w_pose(self,x):
		##x:q_init2,base2_p,base2_w,theta_0
		q_init2=x[:6]
		base2_p=x[6:9]
		base2_w=x[9:-1]
		base2_theta=np.linalg.norm(base2_w)
		base2_k=base2_w/base2_theta
		base2_R=rot(base2_k,base2_theta)

		self.robot2.base_H=H_from_RT(base2_R,base2_p)
		pose2_world_now=self.robot2.fwd(q_init2,world=True)


		R_temp=direction2R(pose2_world_now.R@self.curve_normal[0],-self.curve[1]+self.curve[0])
		R=np.dot(R_temp,Rz(x[-1]))
		try:
			q_init1=self.robot1.inv(pose2_world_now.p,R)[0]
			q_out1,q_out2=self.dual_arm_stepwise_optimize(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,w1=0.02,w2=0.01)
		except:
			# traceback.print_exc()
			return 999

		###make sure extension possible by checking start & end configuration
		if np.min(self.robot1.upper_limit-q_out1[0])<0.2 or  np.min(q_out1[0]-self.robot1.lower_limit)<0.2 or np.min(self.robot1.upper_limit-q_out1[-1])<0.2 or np.min(q_out1[-1]-self.robot1.lower_limit)<0.2:
			return 999

		###make sure extension possible by checking start & end configuration
		if np.min(self.robot2.upper_limit-q_out2[0])<0.2 or  np.min(q_out2[0]-self.robot2.lower_limit)<0.2 or np.min(self.robot2.upper_limit-q_out2[-1])<0.2 or  np.min(q_out2[-1]-self.robot2.lower_limit)<0.2:
			return 999

		speed,_,_=traj_speed_est_dual(self.robot1,self.robot2,q_out1,q_out2,self.lam,self.v_cmd)

		return -min(speed)

	def dual_arm_opt_w_pose_3dof(self,x):
		##x:q_init2,base2_x,base2_y,base2_theta,theta_0
		q_init2=x[:6]
		base2_p=[x[6],x[7],770.5]		###fixed z height
		base2_theta=x[8]
		base2_R=Rz(base2_theta)

		self.robot2.base_H=H_from_RT(base2_R,base2_p)
		pose2_world_now=self.robot2.fwd(q_init2,world=True)

		R_temp=direction2R(pose2_world_now.R@(self.curve_normal[0]),pose2_world_now.R@(-self.curve[1]+self.curve[0]))
		R=np.dot(R_temp,Rz(x[-1]))
		try:
			q_init1=self.robot1.inv(pose2_world_now.R@self.curve[0]+pose2_world_now.p,R)[0]
			q_out1,q_out2,j_out1,j_out2=self.dual_arm_stepwise_optimize(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,w1=0.01,w2=0.01)
		except:
			# traceback.print_exc()
			return 999

		joint_margin=0.2
		
		###make sure extension possible by checking start & end configuration
		if np.min(self.robot1.upper_limit-q_out1[0])<joint_margin or  np.min(q_out1[0]-self.robot1.lower_limit)<joint_margin or np.min(self.robot1.upper_limit-q_out1[-1])<joint_margin or np.min(q_out1[-1]-self.robot1.lower_limit)<joint_margin:
			return 999
		###make sure extension possible by checking start & end configuration
		if np.min(self.robot2.upper_limit-q_out2[0])<joint_margin or  np.min(q_out2[0]-self.robot2.lower_limit)<joint_margin or np.min(self.robot2.upper_limit-q_out2[-1])<joint_margin or  np.min(q_out2[-1]-self.robot2.lower_limit)<joint_margin:
			return 999
		
		# jacobian_margin=0.1
		# jac_check_count=int(len(j_out1)/100)
		# ### J1
		# for J in j_out1[::jac_check_count]:
		# 	_,sv,_=np.linalg.svd(J)
		# 	if np.min(sv)<jacobian_margin:
		# 		# print("min svd of J1 too small")
		# 		return 999
		# _,sv,_=np.linalg.svd(j_out1[-1])
		# if np.min(sv)<jacobian_margin:
		# 	# print("min svd of J1 too small")
		# 	return 999
		# ### J2
		# for J in j_out2[::jac_check_count]:
		# 	_,sv,_=np.linalg.svd(J)
		# 	if np.min(sv)<jacobian_margin:
		# 		# print("min svd of J2 too small")
		# 		return 999
		# _,sv,_=np.linalg.svd(j_out2[-1])
		# if np.min(sv)<jacobian_margin:
		# 	# print("min svd of J1 too small")
		# 	return 999

		# speed,_,_=traj_speed_est_dual(self.robot1,self.robot2,q_out1,q_out2,self.lam,self.v_cmd)
		speed=lambdadot_qlambda_dual(self.robot1,self.robot2,q_out1,q_out2,self.lam)

		return -min(speed)

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

	

def main():
	return 

if __name__ == "__main__":
	main()
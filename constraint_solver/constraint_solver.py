import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, shgo, NonlinearConstraint, minimize
from qpsolvers import solve_qp

sys.path.append('../../toolbox')
from robot_def import *
from lambda_calc import *

class lambda_opt(object):
	def __init__(self,curve,curve_normal,base2_R=np.eye(3),base2_p=np.zeros(3),steps=32):
		self.curve=curve
		self.curve_normal=curve_normal
		self.base2_R=base2_R
		self.base2_p=base2_p
		self.joint_vel_limit=np.radians([110,90,90,150,120,235])
		self.joint_upper_limit=np.radians([220.,160.,70.,300.,120.,360.])
		self.joint_lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.])

		###decrease curve density to simplify computation
		num_per_step=int(len(self.curve)/steps)	
		self.curve=self.curve[0:-1:num_per_step]
		self.curve_normal=self.curve_normal[0:-1:num_per_step]


		###find path length
		self.lam=[0]
		for i in range(len(curve)-1):
			self.lam.append(self.lam[-1]+np.linalg.norm(curve[i+1]-curve[i]))
		###normalize lam, 
		self.lam=np.array(self.lam)/self.lam[-1]
		self.lam=self.lam[0:-1:num_per_step]


	def direction2R(self,v_norm,v_tang):
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

	def normalize_dq(self,q):
		q=q/(np.linalg.norm(q)) 
		return q

	def single_arm_stepwise_optimize(self,q_init,curve=[],curve_normal=[]):
		if len(curve)==0:
			curve=self.curve
			curve_normal=self.curve_normal
		q_all=[q_init]
		q_out=[q_init]
		for i in range(len(curve)):
			try:
				error_fb=999
				if i==0:
					continue

				while error_fb>0.1:

					pose_now=fwd(q_all[-1])
					error_fb=np.linalg.norm(pose_now.p-curve[i])+np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
					
					w=0.2
					Kq=.01*np.eye(6)    #small value to make sure positive definite
					KR=np.eye(3)        #gains for position and orientation error

					J=jacobian(q_all[-1])        #calculate current Jacobian
					Jp=J[3:,:]
					JR=J[:3,:] 
					H=np.dot(np.transpose(Jp),Jp)+Kq+w*np.dot(np.transpose(JR),JR)
					H=(H+np.transpose(H))/2

					vd=curve[i]-pose_now.p
					k=np.cross(pose_now.R[:,-1],curve_normal[i])

					k=k/np.linalg.norm(k)
					theta=-np.arctan2(np.linalg.norm(np.cross(pose_now.R[:,-1],curve_normal[i])), np.dot(pose_now.R[:,-1],curve_normal[i]))

					k=np.array(k)
					s=np.sin(theta/2)*k         #eR2
					wd=-np.dot(KR,s)
					f=-np.dot(np.transpose(Jp),vd)-w*np.dot(np.transpose(JR),wd)
					qdot=solve_qp(H,f)

					q_all.append(q_all[-1]+qdot)
					# print(q_all[-1])
			except:
				q_out.append(q_all[-1])
				traceback.print_exc()
				raise AssertionError
				break

			q_out.append(q_all[-1])

		q_out=np.array(q_out)
		return q_out

	def dual_arm_stepwise_optimize(self,q_init1,q_init2):
		#curve_normal: expressed in second robot tool frame
		q_all1=[q_init1]
		q_out1=[q_init1]
		q_all2=[q_init2]
		q_out2=[q_init2]

		for i in range(len(self.curve)):
			try:
				error_fb=999
				if i==0:
					pose1_now=fwd(q_all1[-1])
					pose2_now=fwd(q_all2[-1])

					pose2_world_now=fwd(q_all2[-1],self.base2_R,self.base2_p)					
					continue


				while error_fb>0.1:
					pose1_now=fwd(q_all1[-1])
					pose2_now=fwd(q_all2[-1])

					pose2_world_now=fwd(q_all2[-1],self.base2_R,self.base2_p)

					error_fb=np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])	

					# print(i)
					# print(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p)-self.curve[i])
					# print(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-self.curve_normal[i])
					########################################################first robot###########################################
					w=0.2
					Kq=.01*np.eye(6)    #small value to make sure positive definite
					KR=np.eye(3)        #gains for position and orientation error
					Kp=0.1				#gains for vd

					J1=jacobian(q_all1[-1])        #calculate current Jacobian
					J1p=J1[3:,:]
					J1R=J1[:3,:] 
					H=np.dot(np.transpose(J1p),J1p)+Kq+w*np.dot(np.transpose(J1R),J1R)
					H=(H+np.transpose(H))/2
					
					vd=Kp*np.dot(pose2_world_now.R,self.curve[i]-np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))  ###R^2_0*(curve-R^0_2*(p1-p2))
					k=np.dot(pose2_world_now.R,np.cross(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1]),self.curve_normal[i]))	###R^2_0*(R^0_2*R1[z] x curve_normal)

					k=k/np.linalg.norm(k)

					vec1=np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])
					theta=-np.arctan2(np.linalg.norm(np.cross(vec1,self.curve_normal[i])),np.dot(vec1,self.curve_normal[i]))

					k=np.array(k)
					s=np.sin(theta/2)*k         #eR2
					wd=-np.dot(KR,s)
					f=-np.dot(np.transpose(J1p),vd)-w*np.dot(np.transpose(J1R),wd)
					qdot=solve_qp(H,f)

					###TODO:
					#cap to joint limits
					q_all1.append(q_all1[-1]+qdot)

					########################################################Second robot###########################################
					J2=jacobian(q_all2[-1])        #calculate current Jacobian, mapped to robot1 base frame
					J2p=np.dot(self.base2_R,J2[3:,:])
					J2R=np.dot(self.base2_R,J2[:3,:] )
					H=np.dot(np.transpose(J2p),J2p)+Kq+w*np.dot(np.transpose(J2R),J2R)
					H=(H+np.transpose(H))/2

					vd=-vd
					theta=-theta

					k=np.array(k)
					s=np.sin(theta/2)*k         #eR2
					wd=-np.dot(KR,s)
					f=-np.dot(np.transpose(J2p),vd)-w*np.dot(np.transpose(J2R),wd)
					qdot=solve_qp(H,f)

					###TODO:
					#cap to joint limits
					q_all2.append(q_all2[-1]+qdot)

			except:
				q_out1.append(q_all1[-1])
				q_out2.append(q_all2[-1])
				traceback.print_exc()
				raise AssertionError
				break

			q_out1.append(q_all1[-1])
			q_out2.append(q_all2[-1])

		q_out1=np.array(q_out1)
		q_out2=np.array(q_out2)
		return q_out1, q_out2

	def curve_pose_opt(self,x):
		k=x[:3]/np.linalg.norm(x[:3])	###pose rotation axis
		theta0=x[3]		###pose rotation angle
		shift=x[4:-1]
		theta1=x[-1]	###spray angle

		R_curve=rot(k,theta0)
		curve_new=np.dot(R_curve,self.curve.T).T+np.tile(shift,(len(self.curve),1))
		curve_normal_new=np.dot(R_curve,self.curve_normal.T).T

		R_temp=self.direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
		R=np.dot(R_temp,Rz(theta1))
		try:
			q_init=inv(curve_new[0],R)[0]
			q_out=self.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
		except:
			# traceback.print_exc()
			return 999

		
		dlam=calc_lamdot(q_out,self.lam,self.joint_vel_limit,1)
		print(min(dlam))
		return -min(dlam)

	def dual_arm_opt(self,x):
		q_init2=x[:-1]

		pose2_world_now=fwd(q_init2,self.base2_R,self.base2_p)

		R_temp=self.direction2R(np.dot(pose2_world_now.R,self.curve_normal[0]),-self.curve[1]+self.curve[0])
		R=np.dot(R_temp,Rz(x[-1]))
		try:
			q_init1=inv(pose2_world_now.p,R)[0]
			q_out1,q_out2=self.dual_arm_stepwise_optimize(q_init1,q_init2)
		except:
			# traceback.print_exc()
			return 999

		
		dlam=calc_lamdot(np.hstack((q_out1,q_out2)),self.lam[:len(q_out2)],np.tile(self.joint_vel_limit,2),1)
		print(min(dlam))
		return -min(dlam)

	def single_arm_theta0_opt(self,theta0):

		R_temp=self.direction2R(self.curve_normal[0],-self.curve[1]+self.curve[0])
		R=np.dot(R_temp,Rz(theta0[0]))
		try:
			q_init=inv(self.curve[0],R)[0]
			q_out=self.single_arm_stepwise_optimize(q_init)
		except:
			return 999

		
		dlam=calc_lamdot(q_out,self.lam,self.joint_vel_limit,1)
		print(min(dlam))
		return -min(dlam)

	def single_arm_global_opt(self,theta):

		for i in range(len(self.curve)):
			if i==0:
				R_temp=self.direction2R(self.curve_normal[i],-self.curve[i+1]+self.curve[i])
				R=np.dot(R_temp,Rz(theta[i]))
				try:
					q_out=[inv(self.curve[i],R)[0]]
				except:
					traceback.print_exc()
					return 999

			else:
				R_temp=self.direction2R(self.curve_normal[i],-self.curve[i]+self.curve[i-1])
				R=np.dot(R_temp,Rz(theta[i]))
				try:
					###get closet config to previous one
					q_inv_all=inv(self.curve[i],R)
					temp_q=q_inv_all-q_out[-1]
					order=np.argsort(np.linalg.norm(temp_q,axis=1))
					q_out.append(q_inv_all[order[0]])
				except:
					# traceback.print_exc()
					return 999


		dlam=calc_lamdot(q_out,self.lam,self.joint_vel_limit,1)
		print(min(dlam))
		return -min(dlam)	

	def calc_js_from_theta(self,theta,curve,curve_normal):

		for i in range(len(curve)):
			if i==0:
				R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
				R=np.dot(R_temp,Rz(theta[i]))
				try:
					q_out=[inv(curve[i],R)[0]]
				except:
					traceback.print_exc()
					return 999

			else:
				R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
				R=np.dot(R_temp,Rz(theta[i]))
				try:
					###get closet config to previous one
					q_inv_all=inv(curve[i],R)
					temp_q=q_inv_all-q_out[-1]
					order=np.argsort(np.linalg.norm(temp_q,axis=1))
					q_out.append(q_inv_all[order[0]])
				except:
					# traceback.print_exc()
					return 999


		return calc_lamdot(q_out,self.lam,self.joint_vel_limit,1)

def main():
	return 

if __name__ == "__main__":
	main()
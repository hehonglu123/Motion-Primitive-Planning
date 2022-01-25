from general_robotics_toolbox import *
import numpy as np

def Rx(theta):
	return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])



ex=np.array([[1],[0],[0]])
ey=np.array([[0],[1],[0]])
ez=np.array([[0],[0],[1]])

###ABB IRB 6650S 125/3.5 Robot Definition
H=np.concatenate((ez,ey,ey,ex,ey,ex),axis=1)
p0=np.array([[0],[0],[0.63]])
p1=np.array([[0.6],[0],[0]])
p2=np.array([[0.],[0],[1.28]])
p3=np.array([[0],[0],[0.2]])
p4=np.array([[1.592],[0],[0]])
p5=np.array([[0.2],[0],[0]])
p6=np.array([[0.0],[0],[0.0]])
#TCP Paint gun
R_tool=Ry(np.radians(120))
p_tool=np.array([0.45,0,-0.05])*1000.
# R_tool=None
# p_tool=None

P=np.concatenate((p0,p1,p2,p3,p4,p5,p6),axis=1)*1000.
joint_type=np.zeros(6)
upper_limit=np.radians([220.,160.,70.,300.,120.,360.])
lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.])
joint_vel_limit=np.radians([110,90,90,150,120,235])
joint_acc_limit=10*joint_vel_limit
ABB_def=Robot(H,P,joint_type,joint_lower_limit = lowerer_limit, joint_upper_limit = upper_limit, joint_vel_limit=joint_vel_limit, R_tool=R_tool,p_tool=p_tool)

class Transform_all(object):
	def __init__(self, p_all, R_all):
		self.R_all=np.array(R_all)
		self.p_all=np.array(p_all)
def jacobian(q):
	return robotjacobian(ABB_def,q)
def fwd(q,base_R=np.eye(3),base_p=np.array([0,0,0])):
	pose_temp=fwdkin(ABB_def,q)
	pose_temp.p=np.dot(base_R,pose_temp.p)+base_p
	pose_temp.R=np.dot(base_R,pose_temp.R)
	return pose_temp

def fwd_all(q_all,base_R=np.eye(3),base_p=np.array([0,0,0])):
	pose_p_all=[]
	pose_R_all=[]
	for q in q_all:
		pose_temp=fwd(q,base_R,base_p)
		pose_p_all.append(pose_temp.p)
		pose_R_all.append(pose_temp.R)

	return Transform_all(pose_p_all,pose_R_all)

def inv(p,R=np.eye(3)):
	pose=Transform(R,p)
	q_all=robot6_sphericalwrist_invkin(ABB_def,pose)
	return q_all
def HomogTrans(q,h,p,jt):

	if jt==0:
		H=np.vstack((np.hstack((rot(h,q), p.reshape((3,1)))),np.array([0, 0, 0, 1,])))
	else:
		H=np.vstack((np.hstack((np.eye(3), p + np.dot(q, h))),np.array([0, 0, 0, 1,])))
	return H
def Hvec(h,jtype):

	if jtype>0:
		H=np.vstack((np.zeros((3,1)),h))
	else:
		H=np.vstack((h.reshape((3,1)),np.zeros((3,1))))
	return H
def phi(R,p):

	Phi=np.vstack((np.hstack((R,np.zeros((3,3)))),np.hstack((-np.dot(R,hat(p)),R))))
	return Phi


def jdot(q,qdot):
	zv=np.zeros((3,1))
	H=np.eye(4)
	J=[]
	Jdot=[]
	n=6
	Jmat=[]
	Jdotmat=[]
	for i in range(n+1):
		if i<n:
			hi=ABB_def.H[:,i]
			qi=q[i]
			qdi=qdot[i]
			ji=ABB_def.joint_type[i]

		else:
			qi=0
			qdi=0
			di=0
			ji=0

		Pi=ABB_def.P[:,i]
		Hi=HomogTrans(qi,hi,Pi,ji)
		Hn=np.dot(H,Hi)
		H=Hn

		PHI=phi(Hi[:3,:3].T,Hi[:3,-1])
		Hveci=Hvec(hi,ji)
		###Partial Jacobian progagation
		if(len(J)>0):
			Jn=np.hstack((np.dot(PHI,J), Hveci))
			temp=np.vstack((np.hstack((hat(hi), np.zeros((3,3)))),np.hstack((np.zeros((3,3)),hat(hi)))))
			Jdotn=-np.dot(qdi,np.dot(temp,Jn)) + np.dot(PHI,np.hstack((Jdot, np.zeros(Hveci.shape))))
		else:
			Jn=Hveci
			Jdotn=np.zeros(Jn.shape)

		Jmat.append(Jn) 
		Jdotmat.append(Jdotn)
		J=Jn
		Jdot=Jdotn

	Jmat[-1]=Jmat[-1][:,:n]
	Jdotmat[-1]=Jdotmat[-1][:,:n]
	return Jdotmat[-1]

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
p_tool=np.array([0.45,0,-0.05])


P=np.concatenate((p0,p1,p2,p3,p4,p5,p6),axis=1)
joint_type=np.zeros(6)
upper_limit=np.radians([220.,160.,70.,300.,120.,360.])
lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.])
joint_vel_limit=np.radians([110,90,90,150,120,235])
ABB_def=Robot(H,P,joint_type,joint_lower_limit = lowerer_limit, joint_upper_limit = upper_limit, joint_vel_limit=joint_vel_limit, R_tool=None,p_tool=None)


def jacobian(q):
	return robotjacobian(ABB_def,q)
def fwd(q):
    return fwdkin(ABB_def,q)

def inv(p,R=np.eye(3)):
	pose=Transform(R,p)
	q_all=robot6_sphericalwrist_invkin(ABB_def,pose)
	return q_all


print(jacobian(np.zeros(6)))
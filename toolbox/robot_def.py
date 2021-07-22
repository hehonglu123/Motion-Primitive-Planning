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


H=np.concatenate((ez,ey,ey,ex,ey,ex),axis=1)
p0=np.array([[0],[0],[0.63]])
p1=np.array([[0.6],[0],[0]])
p2=np.array([[0.],[0],[1.28]])
p3=np.array([[0],[0],[0.2]])
p4=np.array([[1.592],[0],[0]])
p5=np.array([[0.2],[0],[0]])
p6=np.array([[0.0],[0],[0.0]])
#TCP Paint gun
# R_tool=Ry(np.radians(120))
# p_tool=np.array([0.45,0,-0.05])
R_tool=None
p_tool=None

P=np.concatenate((p0,p1,p2,p3,p4,p5,p6),axis=1)
joint_type=np.zeros(6)
upper_limit=np.array([2.967,2.269,1.222,4.712,2.269,6.283])
lowerer_limit=np.array([-2.967,-1.745,-3.491,-4.712,-2.269,-6.283])
ABB_def=Robot(H,P,joint_type,joint_lower_limit = lowerer_limit, joint_upper_limit = upper_limit, R_tool=R_tool,p_tool=p_tool)



def fwd(q):
    return fwdkin(ABB_def,q)

def inv(p,R=np.eye(3)):
	pose=Transform(R,p)
	q_all=robot6_sphericalwrist_invkin(ABB_def,pose)
	return q_all

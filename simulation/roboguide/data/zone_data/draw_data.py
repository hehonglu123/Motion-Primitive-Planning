from general_robotics_toolbox import *
import numpy as np
import yaml
import csv
import matplotlib.pyplot as plt

def Rx(theta):
	return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
ex=np.array([[1],[0],[0]])
ey=np.array([[0],[1],[0]])
ez=np.array([[0],[0],[1]])

#ALL in mm
class m710ic(object):
	#default tool
	def __init__(self,R_tool=Ry(np.radians(90)),p_tool=np.array([0.0,0,0])*1000.,d=0):
		###FANUC m710ic 70 Robot Definition
		self.H=np.concatenate((ez,ey,-ey,-ex,-ey,-ex),axis=1)
		p0=np.array([[0],[0],[0.565]])
		p1=np.array([[0.15],[0],[0]])
		p2=np.array([[0.],[0],[0.87]])
		p3=np.array([[0],[0],[0.17]])   
		p4=np.array([[1.016],[0],[0]])
		p5=np.array([[0.175],[0],[0]])
		p6=np.array([[0.0],[0],[0.0]])

		###fake link for fitting
		tcp_new=p_tool+np.dot(R_tool,np.array([0,0,d]))


		self.P=np.concatenate((p0,p1,p2,p3,p4,p5,p6),axis=1)*1000.
		self.joint_type=np.zeros(6)
		
		###updated range&vel limit
		self.upper_limit=np.radians([170.,85.,70.,300.,120.,360.])
		self.lowerer_limit=np.radians([-170.,-65.,-180.,-300.,-120.,-360.])
		self.joint_vel_limit=np.radians([100,90,90,190,140,190])
		self.joint_acc_limit=10*self.joint_vel_limit
		self.robot_def=Robot(self.H,self.P,self.joint_type,joint_lower_limit = self.lowerer_limit, joint_upper_limit = self.upper_limit, joint_vel_limit=self.joint_vel_limit, R_tool=R_tool,p_tool=tcp_new)

	def jacobian(self,q):
		return robotjacobian(self.robot_def,q)
	def fwd(self,q,base_R=np.eye(3),base_p=np.array([0,0,0])):
		pose_temp=fwdkin(self.robot_def,q)
		pose_temp.p=np.dot(base_R,pose_temp.p)+base_p
		pose_temp.R=np.dot(base_R,pose_temp.R)
		return pose_temp

	def fwd_all(self,q_all,base_R=np.eye(3),base_p=np.array([0,0,0])):
		pose_p_all=[]
		pose_R_all=[]
		for q in q_all:
			pose_temp=fwd(q,base_R,base_p)
			pose_p_all.append(pose_temp.p)
			pose_R_all.append(pose_temp.R)

		return Transform_all(pose_p_all,pose_R_all)

	def inv(self,p,R=np.eye(3)):
		pose=Transform(R,p)
		q_all=robot6_sphericalwrist_invkin(self.robot_def,pose)
		return q_all

robot = m710ic()
print(np.degrees([.1048, -.1981, -.4898, -.0000, -1.0810, -.1048]))
T = robot.fwd([.1048, -.1981, -.4898, -.0000, -1.0810, -.1048])
print(T)

colormap=['red','orange','yellow','green','cyan','blue','purple']
color_i = 0
plt.figure()
ax = plt.axes(projection='3d')
curve_x=np.array([])
curve_y=np.array([])
curve_z=np.array([])
for filename in ['fine','cnt0','cnt25','cnt50','cnt75','cnt100']:
    with open(filename+".DT","r") as f:
        rows = csv.reader(f, delimiter=',')

        log_results_dict = {}
        for col in rows:
            if len(log_results_dict) == 0:
                log_results_dict['timestamp']=[]
                log_results_dict['joint_angle']=[]
                continue
            log_results_dict['timestamp'].append(float(col[0]))
            log_results_dict['joint_angle'].append(np.array([float(col[1]),float(col[2]),float(col[3]),float(col[4]),float(col[5]),float(col[6])]))
        stamps = log_results_dict['timestamp']
        joint_angles = log_results_dict['joint_angle']
    
    for i in range(len(joint_angles)):
        T = robot.fwd(joint_angles[i])
        curve_x = np.append(curve_x,T.p[0])
        curve_y = np.append(curve_y,T.p[1])
        curve_z = np.append(curve_z,T.p[2])
    
    print(curve_x)
    # ax.scatter3D(curve_x, curve_y,curve_z, colormap[color_i],s=3)
    ax.plot(curve_x, curve_y, colormap[color_i],ms=3)
    color_i += 1
plt.show()
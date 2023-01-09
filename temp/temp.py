
import numpy as np 
import copy 
from utils import *
from robots_def import *
from MotionSend import *

robot=robot_obj('ABB_1200_5_90','../config/abb_1200_5_90_robot_default_config.yml')
client = MotionProgramExecClient(base_url="http://127.0.0.1:80")

R=np.array([[-1,0,0],
			[0,1,0],
			[0,0,-1]])
p1=np.array([200,0,300])
p2=np.array([800,0,300])
p3=np.array([500,300,300])

q1=robot.inv(p1,R,np.zeros(6))[0]
q2=robot.inv(p2,R,q1)[0]
q3=robot.inv(p3,R,q1)[0]

mp2=MotionProgram()
mp2.WaitTime(1)

def linear(q1,q2):
	mp = MotionProgram()
	mp.MoveAbsJ(jointtarget(np.degrees(q1),[0]*6),v100,fine)
	mp.WaitTime(1)
	pose_end=robot.fwd(q2)
	cf=quadrant(q2,robot)
	t=robtarget(pose_end.p, R2q(pose_end.R), confdata(cf[0],cf[1],cf[2],cf[3]),[0]*6)
	mp.MoveL(t,v100,fine)

	
	client.execute_multimove_motion_program([mp2,mp])

def arc(q1,q3,q2):
	mp = MotionProgram()
	mp.MoveAbsJ(jointtarget(np.degrees(q1),[0]*6),v100,fine)
	mp.WaitTime(1)
	pose_end=robot.fwd(q3)
	cf=quadrant(q3,robot)
	t1=robtarget(pose_end.p, R2q(pose_end.R), confdata(cf[0],cf[1],cf[2],cf[3]),[0]*6)

	pose_end=robot.fwd(q2)
	cf=quadrant(q2,robot)
	t2=robtarget(pose_end.p, R2q(pose_end.R), confdata(cf[0],cf[1],cf[2],cf[3]),[0]*6)

	mp.MoveC(t1,t2,v100,fine)
	client.execute_multimove_motion_program([mp2,mp])

def rotate(q1,theta):
	mp = MotionProgram()
	mp.MoveAbsJ(jointtarget(np.degrees(q1),[0]*6),v100,fine)
	mp.WaitTime(1)
	q1_new=copy.deepcopy(q1)
	q1_new[-1]+=theta
	mp.MoveAbsJ(jointtarget(np.degrees(q1_new),[0]*6),v100,fine)
	client.execute_multimove_motion_program([mp2,mp])


# linear(q1,q2)
# arc(q1,q3,q2)
rotate(q1,np.pi/2)
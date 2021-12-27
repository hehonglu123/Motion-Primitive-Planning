#!/usr/bin/env python3

from RobotRaconteur.Client import *
import math
import numpy as np
import time
import yaml
from pathlib import Path
import os

import sys
from general_robotics_toolbox import R2q,rot	#convert R to quaternion


server=RRN.ConnectService('rr+tcp://localhost:11346/?service=GazeboServer')
w=server.get_worlds(str(server.world_names[0]))
pose_dtype = RRN.GetNamedArrayDType("com.robotraconteur.geometry.Pose", server)

def decomp(H):
	return R2q(H[:3,:3]),H[:3,3]

def initialize(robot_sdf,model_name,H):
	q,d=decomp(H)
	model_pose = np.zeros((1,), dtype = pose_dtype)
	model_pose["orientation"]['w'] = q[0]
	model_pose["orientation"]['x'] = q[1]
	model_pose["orientation"]['y'] = q[2]
	model_pose["orientation"]['z'] = q[3]
	model_pose["position"]['x']=d[0]
	model_pose["position"]['y']=d[1]
	model_pose["position"]['z']=d[2]
	w.insert_model(robot_sdf, model_name, model_pose)

#model name: ur, sawyer, abb,staubli
#read sdf file
model_name="6650s1"
f = open('model/6650s/model.sdf','r')
robot_sdf = f.read()
with open('calibration/6650s1.yaml') as file:
	H = np.array(yaml.load(file)['H'],dtype=np.float64)
initialize(robot_sdf,model_name,H)
#read sdf file
model_name="6650s2"
f = open('model/6650s/model.sdf','r')
robot_sdf = f.read()
with open('calibration/6650s2.yaml') as file:
	H = np.array(yaml.load(file)['H'],dtype=np.float64)
initialize(robot_sdf,model_name,H)


print("Done!")


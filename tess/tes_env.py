#!/usr/bin/env python3
from tesseract_robotics.tesseract_environment import Environment, ChangeJointOriginCommand
from tesseract_robotics import tesseract_geometry
from tesseract_robotics.tesseract_common import Isometry3d, CollisionMarginData
from tesseract_robotics import tesseract_collision
from tesseract_robotics_viewer import TesseractViewer

import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import yaml, time, traceback, threading, sys
import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import fminbound

sys.path.append('../toolbox')
from gazebo_model_resource_locator import GazeboModelResourceLocator
from robots_def import *

#convert 4x4 H matrix to 3x3 H matrix and inverse for mapping obj to robot frame
def H42H3(H):
	H3=np.linalg.inv(H[:2,:2])
	H3=np.hstack((H3,-np.dot(H3,np.array([[H[0][-1]],[H[1][-1]]]))))
	H3=np.vstack((H3,np.array([0,0,1])))
	return H3

class Tess_Visual(object):
	def __init__(self):

		##load calibration parameters
		# with open('calibration/Sawyer.yaml') as file:
		# 	H_Sawyer 	= np.array(yaml.load(file)['H'],dtype=np.float64)
		# with open('calibration/ABB.yaml') as file:
		# 	H_ABB 	= np.array(yaml.load(file)['H'],dtype=np.float64)
		# self.H_Sawyer=H42H3(H_Sawyer)
		# self.H_ABB=H42H3(H_ABB)

		# self.transformation={'sawyer':H_Sawyer,'abb':H_ABB}

		################kinematics tools##################################################
		
		#link and joint names in urdf
		ABB_6640_180_255_joint_names=["ABB_6640_180_255_joint_1","ABB_6640_180_255_joint_1","ABB_6640_180_255_joint_1","ABB_6640_180_255_joint_1","ABB_6640_180_255_joint_1","ABB_6640_180_255_joint_1"]
		ABB_6640_180_255_link_names=["ABB_6640_180_255_link_1","ABB_6640_180_255_link_2","ABB_6640_180_255_link_3","ABB_6640_180_255_link_4","ABB_6640_180_255_link_5","ABB_6640_180_255_link_6"]
		ABB_1200_5_90_joint_names=['ABB_1200_5_90_joint_1','ABB_1200_5_90_joint_2','ABB_1200_5_90_joint_3','ABB_1200_5_90_joint_4','ABB_1200_5_90_joint_5','ABB_1200_5_90_joint_6']
		ABB_1200_5_90_link_names=['ABB_1200_5_90_link_1','ABB_1200_5_90_link_1','ABB_1200_5_90_link_1','ABB_1200_5_90_link_1','ABB_1200_5_90_link_1','ABB_1200_5_90_link_1']

		#Robot dictionaries, all reference by name
		self.robot_name_list=['ABB_6640_180_255','ABB_1200_5_90']
		self.robot_linkname={'ABB_6640_180_255':ABB_6640_180_255_link_names,'ABB_1200_5_90':ABB_1200_5_90_link_names}
		self.robot_jointname={'ABB_6640_180_255':ABB_6640_180_255_joint_names,'ABB_1200_5_90':ABB_1200_5_90_joint_names}
		

		######tesseract environment setup:
		with open("../config/urdf/combined.urdf",'r') as f:
			combined_urdf = f.read()
		with open("../config/urdf/combined.srdf",'r') as f:
			combined_srdf = f.read()

		self.t_env= Environment()
		self.t_env.init(combined_urdf, combined_srdf, GazeboModelResourceLocator())
		self.scene_graph=self.t_env.getSceneGraph()


		#Tesseract reports all GJK/EPA distance within contact_distance threshold
		contact_distance=0.2
		monitored_link_names = self.t_env.getLinkNames()
		self.manager = self.t_env.getDiscreteContactManager()
		self.manager.setActiveCollisionObjects(monitored_link_names)
		self.manager.setCollisionMarginData(CollisionMarginData(contact_distance))


		#######viewer setup, for URDF setup verification in browser @ localhost:8000/#########################
		self.viewer = TesseractViewer()

		self.viewer.update_environment(self.t_env, [0,0,0])

		self.viewer.start_serve_background()

	def update_pose(model_name,H):
		cmd = ChangeJointOriginCommand(model_name+'_pose', Isometry3d(H_Sawyer))
		self.t_env.applyCommand(cmd)

	def check_collision(robot_name,curve_js):

		return 

	#######################################update joint angles in Tesseract Viewer###########################################
	def viewer_joints_update(self,robots_joint):
		joint_names=[]
		joint_values=[]
		for key in robots_joint:
			joint_names.extend(self.robot_jointname[key])
			joint_values.extend(robots_joint[key].tolist())

		self.viewer.update_joint_positions(joint_names, np.array(joint_values))



def main():


	planner_inst=Tess_Visual()				#create obj


	input("Press enter to quit")
	#stop background checker
	

if __name__ == '__main__':
	main()












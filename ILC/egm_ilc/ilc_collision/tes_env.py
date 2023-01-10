#!/usr/bin/env python3
from tesseract_robotics.tesseract_environment import Environment, ChangeJointOriginCommand, MoveJointCommand
from tesseract_robotics import tesseract_geometry
from tesseract_robotics.tesseract_common import Isometry3d, CollisionMarginData, Translation3d, Quaterniond, \
	ManipulatorInfo
from tesseract_robotics import tesseract_collision
from tesseract_robotics_viewer import TesseractViewer

import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import yaml, time, traceback, threading, sys, json
import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import fminbound

from gazebo_model_resource_locator import GazeboModelResourceLocator
from robots_def import *

#convert 4x4 H matrix to 3x3 H matrix and inverse for mapping obj to robot frame
def H42H3(H):
	H3=np.linalg.inv(H[:2,:2])
	H3=np.hstack((H3,-np.dot(H3,np.array([[H[0][-1]],[H[1][-1]]]))))
	H3=np.vstack((H3,np.array([0,0,1])))
	return H3

class Tess_Env(object):
	def __init__(self,urdf_path):

		#link and joint names in urdf
		ABB_6640_180_255_joint_names=["ABB_6640_180_255_joint_1","ABB_6640_180_255_joint_2","ABB_6640_180_255_joint_3","ABB_6640_180_255_joint_4","ABB_6640_180_255_joint_5","ABB_6640_180_255_joint_6"]
		ABB_6640_180_255_link_names=["ABB_6640_180_255_link_1","ABB_6640_180_255_link_2","ABB_6640_180_255_link_3","ABB_6640_180_255_link_4","ABB_6640_180_255_link_5","ABB_6640_180_255_link_6","ABB_6640_180_255_tool"]

		#Robot dictionaries, all reference by name
		self.robot_linkname={'ABB_6640_180_255':ABB_6640_180_255_link_names}
		self.robot_jointname={'ABB_6640_180_255':ABB_6640_180_255_joint_names}
		

		######tesseract environment setup:
		with open(urdf_path+'combined2.urdf','r') as f:
			combined_urdf = f.read()
		with open(urdf_path+'combined2.srdf','r') as f:
			combined_srdf = f.read()

		self.t_env= Environment()
		self.t_env.init(combined_urdf, combined_srdf, GazeboModelResourceLocator())
		self.scene_graph=self.t_env.getSceneGraph()


		#Tesseract reports all GJK/EPA distance within contact_distance threshold
		contact_distance=0.1
		monitored_link_names = self.t_env.getLinkNames()
		self.manager = self.t_env.getDiscreteContactManager()
		self.manager.setActiveCollisionObjects(monitored_link_names)
		self.manager.setCollisionMarginData(CollisionMarginData(contact_distance))


		#######viewer setup, for URDF setup verification in browser @ localhost:8000/#########################
		self.viewer = TesseractViewer()

		self.viewer.update_environment(self.t_env, [0,0,0])

		self.viewer.start_serve_background()

	def update_pose(self,model_name,H):
		###update model pose in tesseract environment
		cmd = ChangeJointOriginCommand(model_name+'_pose', Isometry3d(H))
		self.t_env.applyCommand(cmd)
		#refresh
		self.viewer.update_environment(self.t_env)

	def check_collision_single(self,robot_name,part_name,q):

		self.t_env.setState(self.robot_jointname[robot_name], q)
		env_state = self.t_env.getState()
		self.manager.setCollisionObjectsTransform(env_state.link_transforms)

		result = tesseract_collision.ContactResultMap()
		contacts = self.manager.contactTest(result,tesseract_collision.ContactRequest(tesseract_collision.ContactTestType_ALL))
		result_vector = tesseract_collision.ContactResultVector()
		tesseract_collision.flattenResults(result,result_vector)

		min_d=9
		J2C=None
		d=None
		###iterate all collision instances
		for c in result_vector:
			cond2=(robot_name in c.link_names[0]) and (part_name in c.link_names[1])	#part collision
			cond3=(part_name in c.link_names[0]) and (robot_name in c.link_names[1])	#part collision

			if c.distance<min_d and (cond2 or cond3):
				min_d=copy.deepcopy(c.distance)
				if cond2:
					J2C=self.robot_linkname[robot_name].index(c.link_names[0])
					d=c.nearest_points[1]-c.nearest_points[0]
				elif cond3:
					J2C=self.robot_linkname[robot_name].index(c.link_names[1])
					d=c.nearest_points[0]-c.nearest_points[1]

		return min_d, J2C, d

	def check_collision_traj(self,robot_name,part_name,curve_js):
		###check collision for a single robot, including self collision and collision with part

		###iterate all joints config 
		min_d_all=[]
		J2C_all=[]
		d_all=[]
		for q in curve_js:
			min_d, J2C, d=self.check_collision_single(robot_name,part_name,q)
			min_d_all.append(min_d)
			J2C_all.append(J2C)
			d_all.append(d)

		return min_d_all, J2C_all, d_all

	#######################################update joint angles in Tesseract Viewer###########################################
	def viewer_joints_update(self,robot_name,joints):
		self.viewer.update_joint_positions(self.robot_jointname[robot_name], np.array(joints))


def collision_test():
	t=Tess_Env('../../../config/urdf/')				#create obj


	q=np.array([0,0.5,0.7,1,1,1.])

	# t.viewer_joints_update('ABB_6640_180_255',q)
	curve_js=[q,np.zeros(6)]
	print(t.check_collision_traj('ABB_6640_180_255','ball_link',curve_js))

	input("Press enter to quit")

if __name__ == '__main__':
	collision_test()











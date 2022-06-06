import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

class ilc_toolbox(object):
	def __init__(self,robot,primitives):
		self.robot=robot
		self.primitives=primitives
	def interp_trajectory(self,q_bp,zone):
		curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,self.primitives,self.robot)
		return blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, self.primitives,self.robot,zone=zone)

	def get_q_bp(self,breakpoints,curve_fit_js):
		q_bp=[]
		for i in range(len(self.primitives)):
			if self.primitives[i]=='movej_fit':
				q_bp.append(points_list[i])
			elif self.primitives[i]=='movel_fit':
				q_bp.append(car2js(self.robot,curve_fit_js[breakpoints[i]],np.array(points_list[i]),self.robot.fwd(curve_fit_js[breakpoints[i]]).R)[0])
			else:
				q_bp.append([car2js(self.robot,curve_fit_js[int((breakpoints[i]+breakpoints[i-1])/2)],points_list[i][0],self.robot.fwd(curve_fit_js[int((breakpoints[i]+breakpoints[i-1])/2)]).R)[0]\
					,car2js(self.robot,curve_fit_js[breakpoints[i]],points_list[i][0],self.robot.fwd(curve_fit_js[breakpoints[i]]).R)[0]])
		return q_bp

	def get_gradient_from_model_6(self,q_bp,breakpoints_blended,curve_blended,curve_R_blended,max_error_curve_blended_idx,worst_point_pose,closest_p,closest_N):
		###q_bp:				joint configs at breakpoints
		###breakpoints_blended:	breakpoints of blended trajectory
		###curve_blended:		blended trajectory
		###curve_R_blended:		blended trajectory
		###max_error_curve_blended_idx:	p', closest point to worst case error on blended trajectory
		###worst_point_pose:	execution curve with worst case pose
		###closest_p:			closest point on original curve
		###closest_N:			closest point on original curve

		de_dp=[]    #de_dp1q1,de_dp1q2,...,de_dp3q6
		de_ori_dp=[]
		delta=0.01 	#rad
		###find closest 3 breakpoints
		order=np.argsort(np.abs(breakpoints_blended-max_error_curve_blended_idx))
		breakpoint_interp_2tweak_indices=order[:3]

		###len(primitives)==len(breakpoints)==len(breakpoints_blended)==len(points_list)
		for m in breakpoint_interp_2tweak_indices:  #3 breakpoints
			for n in range(6): #6DOF, q1~q6
				q_bp_temp=copy.deepcopy(q_bp)
				q_bp_temp[m]+=delta
				#restore new trajectory
				curve_interp_temp, curve_R_interp_temp, curve_js_interp_temp, breakpoints_blended_temp=form_traj_from_bp(q_bp_temp,primitives,robot)
				curve_js_blended_temp,curve_blended_temp,curve_R_blended_temp=blend_js_from_primitive(curve_interp_temp, curve_js_interp_temp, breakpoints_blended_temp, primitives,robot,zone=10)
				

				worst_case_point_shift=curve_blended_temp[max_error_curve_blended_idx]-curve_blended[max_error_curve_blended_idx]
				de=np.linalg.norm(worst_point_pose.p+worst_case_point_shift-closest_p)
				worst_case_R_shift=curve_R_blended_temp[max_error_curve_blended_idx]@curve_R_blended[max_error_curve_blended_idx].T
				de_ori=get_angle(worst_case_R_shift@worst_point_pose.R,closest_N)

				de_dp.append(de/delta)
				de_ori_dp.append(de_ori/delta)

		de_dp=np.reshape(de_dp,(-1,1))
		de_ori_dp=np.reshape(de_ori_dp,(-1,1))

		return de_dp, de_ori_dp

	def update_bp_6(self,points_list,q_list,q_bp,de_dp,de_ori_dp,max_error,max_ori_error,breakpoint_interp_2tweak_indices,alpha1=0.5,alpha2=0.1):
		bp_q_adjustment=-alpha1*np.linalg.pinv(de_dp)*max_error-alpha2*np.linalg.pinv(de_ori_dp)*max_ori_error
		for i in range(len(breakpoint_interp_2tweak_indices)):  #3 breakpoints
			q_bp[breakpoint_interp_2tweak_indices[i]]+=bp_q_adjustment[0][6*i:6*(i+1)]
			pose_temp=self.robot.fwd(q_bp[breakpoint_interp_2tweak_indices[i]])
			points_list[breakpoint_interp_2tweak_indices[i]]=pose_temp.p
			q_list[breakpoint_interp_2tweak_indices[i]]=R2q(pose_temp.R)

		return points_list, q_list


	def get_gradient_from_model_xyz(self,q_bp,p_bp,breakpoints_blended,curve_blended,max_error_curve_blended_idx,worst_point_pose,closest_p,breakpoint_interp_2tweak_indices):
		###q_bp:				joint configs at breakpoints
		###p_bp:				xyz configs at breakpoints
		###breakpoints_blended:	breakpoints of blended trajectory
		###curve_blended:		blended trajectory
		###max_error_curve_blended_idx:	p', closest point to worst case error on blended trajectory
		###worst_point_pose:	execution curve with worst case pose
		###closest_p:			closest point on original curve
		###breakpoint_interp_2tweak_indices:	closest N breakpoints

		###TODO:ADD MOVEC SUPPORT
		de_dp=[]    #de_dp1q1,de_dp1q2,...,de_dp3q6
		de_ori_dp=[]
		delta=0.1 	#mm
		

		###len(primitives)==len(breakpoints)==len(breakpoints_blended)==len(points_list)
		for m in breakpoint_interp_2tweak_indices:  #3 breakpoints
			for n in range(3): #3DOF, xyz
				q_bp_temp=np.array(copy.deepcopy(q_bp))
				p_bp_temp=copy.deepcopy(p_bp)
				p_bp_temp[m][n]+=delta

				q_bp_temp[m][0]=car2js(self.robot,q_bp[m][0],np.array(p_bp_temp[m]),self.robot.fwd(q_bp[m][0]).R)[0]###TODO:ADD MOVEC SUPPORT

				#restore new trajectory, only for adjusted breakpoint, 1-bp change requires traj interp from 5 bp
				short_version=range(max(m-2,0),min(m+2,len(breakpoints_blended)-1))
				curve_interp_temp, curve_R_interp_temp, curve_js_interp_temp, breakpoints_blended_temp=form_traj_from_bp(q_bp_temp[short_version],[self.primitives[i] for i in short_version],self.robot)

				curve_js_blended_temp,curve_blended_temp,curve_R_blended_temp=blend_js_from_primitive(curve_interp_temp, curve_js_interp_temp, breakpoints_blended_temp, [self.primitives[i] for i in short_version],self.robot,zone=10)
				
				curve_blended_new=copy.deepcopy(curve_blended)
				start_idx=int((breakpoints_blended[short_version[0]]+breakpoints_blended[short_version[1]])/2)
				end_idx=int((breakpoints_blended[short_version[-1]]+breakpoints_blended[short_version[-2]])/2)

				curve_blended_new[start_idx:end_idx]=curve_blended_temp[start_idx-breakpoints_blended[short_version[0]]:-1-(breakpoints_blended[short_version[-1]]-end_idx)]

				worst_case_point_shift=curve_blended_new[max_error_curve_blended_idx]-curve_blended[max_error_curve_blended_idx]
				###get new error
				de=np.linalg.norm(worst_point_pose.p+worst_case_point_shift-closest_p)-np.linalg.norm(worst_point_pose.p-closest_p)

				de_dp.append(de/delta)

		de_dp=np.reshape(de_dp,(-1,1))

		return de_dp

	def update_bp_xyz(self,p_bp,q_bp,de_dp,max_error,breakpoint_interp_2tweak_indices,alpha=0.5):

		point_adjustment=-alpha*np.linalg.pinv(de_dp)*max_error

		for i in range(len(breakpoint_interp_2tweak_indices)):  #3 breakpoints
			p_bp[breakpoint_interp_2tweak_indices[i]]+=point_adjustment[0][3*i:3*(i+1)]
			###TODO:ADD MOVEC SUPPORT
			q_bp[breakpoint_interp_2tweak_indices[i]][0]=car2js(self.robot,q_bp[breakpoint_interp_2tweak_indices[i]][0],p_bp[breakpoint_interp_2tweak_indices[i]],self.robot.fwd(q_bp[breakpoint_interp_2tweak_indices[i]][0]).R)[0]

		return p_bp, q_bp

	def get_gradient_from_model_ori(self,q_bp,R_bp,breakpoints_blended,curve_R_blended,max_error_curve_blended_idx,worst_point_pose,closest_R):
		###q_bp:				joint configs at breakpoints
		###R_bp:				ori configs at breakpoints
		###breakpoints_blended:	breakpoints of blended trajectory
		###curve_R_blended:		blended trajectory R
		###curve_R_blended:		blended trajectory
		###max_error_curve_blended_idx:	p', closest point to worst case error on blended trajectory
		###worst_point_pose:	execution curve with worst case pose
		###closest_R:			closest point on original curve

		de_dp=[]    #de_dp1q1,de_dp1q2,...,de_dp3q6
		de_ori_dp=[]
		delta=0.01 	#rad
		###find closest 3 breakpoints
		order=np.argsort(np.abs(breakpoints_blended-max_error_curve_blended_idx))
		breakpoint_interp_2tweak_indices=order[:3]

		###len(primitives)==len(breakpoints)==len(breakpoints_blended)==len(points_list)
		for m in breakpoint_interp_2tweak_indices:  #3 breakpoints
			for n in range(3): #3DOF, xyz
				q_bp_temp=copy.deepcopy(q_bp)
				p_bp_temp=copy.deepcopy(p_bp)
				p_bp_temp[m][n]+=delta
				q_bp_temp[m]=car2js(robot,q_bp_temp[m],np.array(q_bp_temp[m]),robot.fwd(q_bp_temp[m]).R)[0]

				#restore new trajectory
				curve_interp_temp, curve_R_interp_temp, curve_js_interp_temp, breakpoints_blended_temp=form_traj_from_bp(q_bp_temp,primitives,robot)
				curve_js_blended_temp,curve_blended_temp,curve_R_blended_temp=blend_js_from_primitive(curve_interp_temp, curve_js_interp_temp, breakpoints_blended_temp, primitives,robot,zone=10)
				

				worst_case_point_shift=curve_blended_temp[max_error_curve_blended_idx]-curve_blended[max_error_curve_blended_idx]
				de=np.linalg.norm(worst_point_pose.p+worst_case_point_shift-closest_p)

				de_dp.append(de/delta)

		de_dp=np.reshape(de_dp,(-1,1))

		return de_dp


	def update_bp_ori(self,points_list,q_list,q_bp,de_dp,de_ori_dp,max_error,max_ori_error,breakpoint_interp_2tweak_indices,alpha1=0.5,alpha2=0.1):
		bp_q_adjustment=-alpha1*np.linalg.pinv(de_dp)*max_error-alpha2*np.linalg.pinv(de_ori_dp)*max_ori_error
		for i in range(len(breakpoint_interp_2tweak_indices)):  #3 breakpoints
			q_bp[breakpoint_interp_2tweak_indices[i]]+=bp_q_adjustment[0][6*i:6*(i+1)]
			pose_temp=self.robot.fwd(q_bp[breakpoint_interp_2tweak_indices[i]])
			points_list[breakpoint_interp_2tweak_indices[i]]=pose_temp.p
			q_list[breakpoint_interp_2tweak_indices[i]]=R2q(pose_temp.R)

		return points_list, q_list


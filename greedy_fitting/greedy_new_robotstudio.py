#############
# This script takes the result of greedy fit 
# and execute the motion primitives in the RobotStudio
# using https://github.com/johnwason/abb_motion_program_exec
# The greedy_fit part is exactly the same as greedy_new.py
#############

import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
import sys
sys.path.append('../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from robot_def import *
from direction2R import *
from general_robotics_toolbox import *
from error_check import *
from robotstudio_send import MotionSend

#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched breakpoints###############################

class greedy_fit(object):
	def __init__(self,curve,curve_normal,d=50):
		self.d=d 			###standoff distance
		self.curve=curve
		self.curve_backproj=curve-self.d*curve_normal
		self.curve_normal=curve_normal

		self.joint_vel_limit=np.radians([110,90,90,150,120,235])
		self.joint_acc_limit=10*self.joint_vel_limit
		self.joint_upper_limit=np.radians([220.,160.,70.,300.,120.,360.])
		self.joint_lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.])

		###seed initial js for inv
		self.q_init=np.array([0.627463700138299,0.17976842821744082,0.5196590573281621,1.6053098733278601,-0.8935105128511388,0.9174696574156079])

		self.curve_fit=[]
		self.curve_fit_R=[]
		self.curve_fit_js=[]
		self.cartesian_slope_prev=None
		
		###slope alignement settings
		self.slope_constraint=False
		self.break_early=False

		###initial primitive candidates
		self.primitives={'movel_fit':self.movel_fit,'movej_fit':self.movej_fit,'movec_fit':self.movec_fit}
	
		###find path length
		self.lam=[0]
		for i in range(len(curve)-1):
			self.lam.append(self.lam[-1]+np.linalg.norm(curve[i+1]-curve[i]))

	def project(self,curve_fit,curve_fit_R):
		###project fitting curve by standoff distance
		curve_fit_proj=curve_fit+self.d*curve_fit_R[:,:,-1]

		return curve_fit_proj

	def orientation_interp(self,R_init,R_end,steps):
		curve_fit_R=[]
		###find axis angle first
		R_diff=np.dot(R_init.T,R_end)
		k,theta=R2rot(R_diff)
		for i in range(steps):
			###linearly interpolate angle
			angle=theta*i/steps
			R=rot(k,angle)
			curve_fit_R.append(np.dot(R_init,R))
		curve_fit_R=np.array(curve_fit_R)
		return curve_fit_R

	def car2js(self,curve_fit,curve_fit_R):
		###calculate corresponding joint configs
		curve_fit_js=[]
		for i in range(len(curve_fit)):
			q_all=np.array(inv(curve_fit[i],curve_fit_R[i]))

			###choose inv_kin closest to previous joints
			temp_q=q_all-self.q_prev
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			curve_fit_js.append(q_all[order[0]])
		return curve_fit_js
	def movel_fit(self,curve,curve_backproj,curve_normal):	###unit vector slope
		###no constraint
		if len(self.curve_fit)==0:
			A=np.vstack((np.ones(len(curve_backproj)),np.arange(0,len(curve_backproj)))).T
			b=curve_backproj
			res=np.linalg.lstsq(A,b,rcond=None)[0]
			start_point=res[0]
			slope=res[1].reshape(1,-1)

		###with constraint point
		else:
			start_point=self.curve_fit[-1]

			if self.slope_constraint:
				
				slope=self.cartesian_slope_prev*np.linalg.norm(curve_backproj[-1]-curve_backproj[0])/len(curve_backproj)
				slope=slope.reshape(1,-1)
			else:

				A=np.arange(0,len(curve_backproj)).reshape(-1,1)
				b=curve_backproj-start_point
				res=np.linalg.lstsq(A,b,rcond=None)[0]
				slope=res.reshape(1,-1)


		curve_fit=np.dot(np.arange(0,len(curve_backproj)).reshape(-1,1),slope)+start_point
		###calculate fitting error
		max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

		###interpolate orientation linearly
		R_end=direction2R(curve_normal[-1],-curve_fit[-1]+curve_fit[-2])
		if len(self.curve_fit)==0:
			R_init=direction2R(curve_normal[0],-curve_fit[1]+curve_fit[0])
		else:
			R_init=self.curve_fit_R[-1]
		curve_fit_R=self.orientation_interp(R_init,R_end,len(curve_fit))


		###calculate corresponding joint configs
		# curve_fit_js=self.car2js(curve_fit,curve_fit_R)
		curve_fit_js=[]

		###calculating projection error
		curve_proj=self.project(curve_fit,curve_fit_R)
		max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
		max_error=(max_error1+max_error2)/2

		# print(max_error1,max_error2)
		return curve_fit,curve_fit_R,curve_fit_js,max_error


	def movej_fit(self,curve,curve_backproj,curve_normal):
		if len(q)==0:
			A=np.vstack((np.ones(len(curve_backproj_js)),np.arange(0,len(curve_backproj_js)))).T
			b=curve_backproj_js
			res=np.linalg.lstsq(A,b,rcond=None)[0]
			start_point=res[0]
			slope=res[1].reshape(1,-1)

			start_pose=fwd(curve_backproj_js[0])
		###with constraint point
		else:
			start_pose=fwd(q)
			A=np.arange(0,len(curve_backproj_js)).reshape(-1,1)
			b=curve_backproj_js-curve_backproj_js[0]
			res=np.linalg.lstsq(A,b,rcond=None)[0]
			start_point=q
			slope=res.reshape(1,-1)

		curve_js_fit=np.dot(np.arange(0,len(curve_backproj_js)).reshape(-1,1),slope)+start_point

		curve_fit=[]
		for i in range(len(curve_js_fit)):
			curve_fit.append(fwd(curve_js_fit[i]).p)
		curve_fit=np.array(curve_fit)

		###calculate fitting error
		max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

		###calculating projection error
		curve_proj=project(curve_fit,start_pose.R,fwd(curve_js_fit[-1]).R)
		max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
		max_error=(max_error1+max_error2)/2

		return curve_fit,curve_js_fit[-1],max_error


	def movec_fit(self,curve,curve_backproj,curve_normal):
		###no previous constraint
		if len(self.curve_fit)==0:
			curve_fit,curve_fit_circle=circle_fit(curve_backproj)	
		###with constraint point
		else:
			if self.slope_constraint:
				curve_fit,curve_fit_circle=circle_fit(curve_backproj,self.curve_fit[-1],self.cartesian_slope_prev)
			else:
				curve_fit,curve_fit_circle=circle_fit(curve_backproj,self.curve_fit[-1])
		
		max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))
		###interpolate orientation linearly
		R_end=direction2R(curve_normal[-1],-curve_fit[-1]+curve_fit[-2])
		if len(self.curve_fit)==0:
			R_init=direction2R(curve_normal[0],-curve_fit[1]+curve_fit[0])
		else:
			R_init=self.curve_fit_R[-1]
		curve_fit_R=self.orientation_interp(R_init,R_end,len(curve_fit))

		###calculate corresponding joint configs
		# curve_fit_js=self.car2js(curve_fit,curve_fit_R)
		curve_fit_js=[]

		###calculating projection error
		curve_proj=self.project(curve_fit,curve_fit_R)
		max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
		max_error=(max_error1+max_error2)/2

		# print(max_error1,max_error2)
		return curve_fit,curve_fit_R,curve_fit_js,max_error

	def breakearly(self,curve,curve_fit):
		best_alignment=0
		idx=0
		for i in reversed(range(len(curve)-1)):
			slope_original=curve[i+1]-curve[i]
			slope_original=slope_original/np.linalg.norm(slope_original)

			slope_fit=curve_fit[i+1]-curve_fit[i]
			slope_fit=slope_fit/np.linalg.norm(slope_fit)

			###return the point with smaller slope difference
			if np.dot(slope_fit,slope_original)>best_alignment:
				best_alignment=np.dot(slope_fit,slope_original)
				idx=i

		print('alignment:',best_alignment)

		if idx==0:
			idx=int(len(curve)/2)
		
		return idx

	def fit_under_error(self,max_error_threshold):

		###initialize
		breakpoints=[1]
		breakpoints_out=[0]
		primitives_choices=[]
		points=[]


		results_max_cartesian_error=[]
		results_max_cartesian_error_index=[]
		results_avg_cartesian_error=[]
		results_max_orientation_error=[]
		results_max_dz_error=[]
		results_avg_dz_error=[]

		self.curve_fit=[]
		self.curve_fit_R=[]
		self.curve_fit_js=[]
		self.cartesian_slope_prev=None

		while breakpoints[-1]<len(self.curve)-1:
			
			
			

			next_point= min(2000,len(self.curve)-1-breakpoints[-1])
			prev_point=0
			prev_possible_point=0

			max_errors={'movel_fit':999,'movej_fit':999,'movec_fit':999}
			###initial error map update:
			for key in self.primitives: 
				curve_fit,curve_fit_R,curve_fit_js,max_error=self.primitives[key](self.curve[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_normal[breakpoints[-1]-1:breakpoints[-1]+next_point])
				max_errors[key]=max_error

			###bisection search breakpoints
			while True:
				print(breakpoints[-1]+next_point,max_errors[min(max_errors, key=max_errors.get)])
				###bp going backward to meet threshold
				if min(list(max_errors.values()))>max_error_threshold:
					prev_point_temp=next_point
					next_point-=int(np.abs(next_point-prev_point)/2)
					prev_point=prev_point_temp
					
					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error=self.primitives[key](self.curve[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_normal[breakpoints[-1]-1:breakpoints[-1]+next_point])
						max_errors[key]=max_error



				###bp going forward to get close to threshold
				else:
					prev_possible_point=next_point
					prev_point_temp=next_point
					next_point= min(next_point + int(np.abs(next_point-prev_point)),len(self.curve)-1-breakpoints[-1])
					prev_point=prev_point_temp
					

					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error=self.primitives[key](self.curve[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_normal[breakpoints[-1]-1:breakpoints[-1]+next_point])
						max_errors[key]=max_error

				# print(max_errors)
				if next_point==prev_point:
					print('stuck')		###if ever getting stuck, restore
					###TODO: debug why <2
					next_point=max(prev_possible_point,2)
					# if breakpoints[-1]+next_point+1==len(self.curve)-1:
					# 	next_point=3

					
					
					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error=self.primitives[key](self.curve[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_normal[breakpoints[-1]-1:breakpoints[-1]+next_point])
						if max_error<max_error_threshold:
							primitives_choices.append(key)
							if key=='movec_fit':
								points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
							elif key=='movel_fit':
								points.append([curve_fit[-1]])
							else:
								points.append([curve_fit_js[-1]])
							break
	
					break

				###find the closest but under max_threshold
				if (min(list(max_errors.values()))<=max_error_threshold and np.abs(next_point-prev_point)<10) or next_point==len(self.curve)-1:
					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error=self.primitives[key](self.curve[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],self.curve_normal[breakpoints[-1]-1:breakpoints[-1]+next_point])
						if max_error<max_error_threshold:
							primitives_choices.append(key)
							if key=='movec_fit':
								points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
							elif key=='movel_fit':
								points.append([curve_fit[-1]])
							else:
								points.append([curve_fit_js[-1]])
							
							break
					break
	

			if self.break_early:
				idx=max(self.breakearly(self.curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_fit),2)
			else:
				idx=next_point

			breakpoints.append(breakpoints[-1]+idx)
			self.curve_fit.extend(curve_fit[:len(curve_fit)-(next_point-idx)])
			self.curve_fit_R.extend(curve_fit_R[:len(curve_fit)-(next_point-idx)])
			# self.curve_fit_js.extend(curve_fit_js[:len(curve_fit)-(next_point-idx)])
			###calculating ending slope
			self.cartesian_slope_prev=(self.curve_fit[-1]-self.curve_fit[-2])/np.linalg.norm(self.curve_fit[-1]-self.curve_fit[-2])
			breakpoints_out.append(breakpoints_out[-1]+len(curve_fit[:len(curve_fit)-(next_point-idx)]))


			print(breakpoints)
			print(primitives_choices)
			# print(points)

		##############################check error (against fitting back projected curve)##############################

		# max_error,max_error_idx=calc_max_error(self.curve_fit,self.curve_backproj)
		# print('max error: ', max_error)

		self.curve_fit=np.array(self.curve_fit)
		self.curve_fit_js=np.array(self.curve_fit_js)

		return breakpoints,breakpoints_out,primitives_choices,points




def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_ge/Curve_in_base_frame.csv", names=col_names)
	# curve_x=data['X'].tolist()
	# curve_y=data['Y'].tolist()
	# curve_z=data['Z'].tolist()
	# curve_direction_x=data['direction_x'].tolist()
	# curve_direction_y=data['direction_y'].tolist()
	# curve_direction_z=data['direction_z'].tolist()
	test_length = 1000
	curve_x=data['X'].tolist()[:test_length]
	curve_y=data['Y'].tolist()[:test_length]
	curve_z=data['Z'].tolist()[:test_length]
	curve_direction_x=data['direction_x'].tolist()[:test_length]
	curve_direction_y=data['direction_y'].tolist()[:test_length]
	curve_direction_z=data['direction_z'].tolist()[:test_length]
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


	greedy_fit_obj=greedy_fit(curve,curve_normal,d=50)

	###disable slope alignment
	greedy_fit_obj.slope_constraint=True
	greedy_fit_obj.break_early=False
	###set primitive choices
	greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit,'movec_fit':greedy_fit_obj.movec_fit}

	breakpoints,breakpoints_out,primitives_choices,points=greedy_fit_obj.fit_under_error(max_error_threshold=1.)


	###plt
	###3D plot
	# plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(greedy_fit_obj.curve[:,0], greedy_fit_obj.curve[:,1],greedy_fit_obj.curve[:,2], 'gray')
	
	# ax.scatter3D(greedy_fit_obj.curve_fit[:,0], greedy_fit_obj.curve_fit[:,1], greedy_fit_obj.curve_fit[:,2], c=greedy_fit_obj.curve_fit[:,2], cmap='Greens')
	# plt.show()

	############insert initial configuration#################
	primitives_choices.insert(0,'movej_fit')
	q_all=np.array(inv(greedy_fit_obj.curve_fit[0],greedy_fit_obj.curve_fit_R[0]))
	###choose inv_kin closest to previous joints
	temp_q=q_all-greedy_fit_obj.q_init
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_init=q_all[order[0]]
	points.insert(0,[q_init])

	print(fwd(q_init))
	print(rox.R2q(fwd(q_init).R))
	print(fwd(np.deg2rad([31.592, 6.91965, 40.4833, 50.5172, -45.0545, 79.0079])))

	ms = MotionSend()
	ms.exec_motions(primitives_choices,breakpoints,points)

	
	print(len(primitives_choices))
	print(len(points))
	print(len(breakpoints_out))
	print(len(breakpoints))

	df=DataFrame({'breakpoints':breakpoints,'breakpoints_out':breakpoints_out,'primitives':primitives_choices,'points':points})
	df.to_csv('command_backproj.csv',header=True,index=False)
	df=DataFrame({'x':greedy_fit_obj.curve_fit[:,0],'y':greedy_fit_obj.curve_fit[:,1],'z':greedy_fit_obj.curve_fit[:,2]})
	df.to_csv('curve_fit_backproj.csv',header=True,index=False)

if __name__ == "__main__":
	main()

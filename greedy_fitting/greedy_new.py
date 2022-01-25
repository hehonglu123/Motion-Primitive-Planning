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

#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched breakpoints###############################

class greedy_fit(object):
	def __init__(self,curve,curve_normal):
		self.d=50 			###standoff distance
		self.curve=curve
		self.curve_backproj=curve-self.d*curve_normal
		self.curve_normal=curve_normal
		self.base2_R=base2_R
		self.base2_p=base2_p
		self.joint_vel_limit=np.radians([110,90,90,150,120,235])
		self.joint_acc_limit=10*self.joint_vel_limit
		self.joint_upper_limit=np.radians([220.,160.,70.,300.,120.,360.])
		self.joint_lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.])

		###seed initial js for inv
		self.q_prev=np.array([0.627463700138299,0.17976842821744082,0.5196590573281621,1.6053098733278601,-0.8935105128511388,0.9174696574156079])

		self.curve_fit=[]
		self.curve_fit_R=[]
		self.curve_fit_js=[]
		self.cartesian_slope_prev=None
	
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
		for i in range(steps):
			###linearly interpolate angle
			angle=theta*i/steps
			R=rot(k,angle)
			curve_fit_R.append(np.dot(R_init,R))
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
			slope=self.cartesian_slope_prev*np.linalg.norm(curve_backproj[-1]-curve_backproj[0])/len(curve_backproj)

		curve_fit=np.dot(np.arange(0,len(curve_backproj)).reshape(-1,1),slope)+start_point
		###calculate fitting error
		max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

		###interpolate orientation linearly
		R_end=direction2R(curve_normal[-1],-curve_fit[-1]+curve_fit[-2])
		if len(self.curve_fit)==0:
			R_init=direction2R(curve_normal[0],-curve_fit[1]+curve_fit[0])
		else:
			R_init=self.curve_fit_R[-1]
		curve_fit_R=orientation_interp(R_init,R_end,len(curve_fit))


		###calculate corresponding joint configs
		curve_fit_js=self.car2js(curve_fit,curve_fit_R)

		###calculating projection error
		curve_proj=project(curve_fit,curve_fit_R)
		max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
		max_error=(max_error1+max_error2)/2

		# print(max_error1,max_error2)
		return curve_fit,curve_fit_R,curve_fit_js,max_error,slope


	# def movej_fit(self,curve,curve_backproj,curve_normal):
	# 	if len(q)==0:
	# 		A=np.vstack((np.ones(len(curve_backproj_js)),np.arange(0,len(curve_backproj_js)))).T
	# 		b=curve_backproj_js
	# 		res=np.linalg.lstsq(A,b,rcond=None)[0]
	# 		start_point=res[0]
	# 		slope=res[1].reshape(1,-1)

	# 		start_pose=fwd(curve_backproj_js[0])
	# 	###with constraint point
	# 	else:
	# 		start_pose=fwd(q)
	# 		A=np.arange(0,len(curve_backproj_js)).reshape(-1,1)
	# 		b=curve_backproj_js-curve_backproj_js[0]
	# 		res=np.linalg.lstsq(A,b,rcond=None)[0]
	# 		start_point=q
	# 		slope=res.reshape(1,-1)

	# 	curve_js_fit=np.dot(np.arange(0,len(curve_backproj_js)).reshape(-1,1),slope)+start_point

	# 	curve_fit=[]
	# 	for i in range(len(curve_js_fit)):
	# 		curve_fit.append(fwd(curve_js_fit[i]).p)
	# 	curve_fit=np.array(curve_fit)

	# 	###calculate fitting error
	# 	max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

	# 	###calculating projection error
	# 	curve_proj=project(curve_fit,start_pose.R,fwd(curve_js_fit[-1]).R)
	# 	max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
	# 	max_error=(max_error1+max_error2)/2

	# 	return curve_fit,curve_js_fit[-1],max_error


	def movec_fit(self,curve,curve_backproj,curve_normal):
		###no previous constraint
		if len(self.curve_fit)==0:
			curve_fit,curve_fit_circle=circle_fit(curve_backproj)	
		###with constraint point
		else:
			curve_fit,curve_fit_circle=circle_fit(curve_backproj,self.curve_fit[-1],self.cartesian_slope_prev)
		
		max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

		###interpolate orientation linearly
		R_end=direction2R(curve_normal[-1],-curve_fit[-1]+curve_fit[-2])
		if len(self.curve_fit)==0:
			R_init=direction2R(curve_normal[0],-curve_fit[1]+curve_fit[0])
		else:
			R_init=self.curve_fit_R[-1]
		curve_fit_R=orientation_interp(R_init,R_end,len(curve_fit))


		###calculate corresponding joint configs
		curve_fit_js=self.car2js(curve_fit,curve_fit_R)

		###calculating projection error
		curve_proj=project(curve_fit,curve_fit_R)
		max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
		max_error=(max_error1+max_error2)/2

		# print(max_error1,max_error2)
		return curve_fit,curve_fit_R,curve_fit_js,max_error,slope



	def fit_under_error(curve,curve_backproj,curve_backproj_js,max_error_threshold,d=50):

		###initialize
		breakpoints=[1]
		breakpoints_out=[0]
		primitives_choices=[]
		q_breakpoints=[]
		points=[]


		results_max_cartesian_error=[]
		results_max_cartesian_error_index=[]
		results_avg_cartesian_error=[]
		results_max_orientation_error=[]
		results_max_dz_error=[]
		results_avg_dz_error=[]

		fit=[]
		js_fit=[]

		while breakpoints[-1]!=len(curve)-1:
			###primitive candidates
			primitives={'movel_fit':movel_fit,'movej_fit':movej_fit,'movec_fit':movec_fit}
			
			

			next_point= min(2000,len(curve)-1-breakpoints[-1])
			prev_point=0
			prev_possible_point=0

			max_errors={'movel_fit':999,'movej_fit':999,'movec_fit':999}

			###bisection search breakpoints
			while True:
				###bp going backward to meet threshold
				if min(list(max_errors.values()))>max_error_threshold:
					prev_point_temp=next_point
					next_point-=int(np.abs(next_point-prev_point)/2)
					prev_point=prev_point_temp
					
					for key in primitives: 
						curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]-1:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
						max_errors[key]=max_error



				###bp going forward to get close to threshold
				else:
					prev_possible_point=next_point
					prev_point_temp=next_point
					next_point= min(next_point + int(np.abs(next_point-prev_point)),len(curve)-1-breakpoints[-1])
					prev_point=prev_point_temp
					

					for key in primitives: 
						curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]-1:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
						max_errors[key]=max_error

				# print(max_errors)
				if next_point==prev_point:
					print('stuck')
					###if ever getting stuck, restore
					next_point=prev_possible_point
					
					
					for key in primitives: 
						curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]-1:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
						if max_error<max_error_threshold:
							q_breakpoints.append(q_last)
							primitives_choices.append(key)
							if key=='movec_fit':
								points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
							elif key=='movel_fit':
								points.append([curve_fit[-1]])
							else:
								points.append([q_last])
							break

					

					breakpoints.append(breakpoints[-1]+next_point)
					fit.append(curve_fit)
					breakpoints_out.append(breakpoints_out[-1]+len(curve_fit))
					break

				###find the closest but under max_threshold
				if (min(list(max_errors.values()))<=max_error_threshold and np.abs(next_point-prev_point)<10) or next_point==len(curve)-1:
					for key in primitives: 
						curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]-1:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]-1:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
						if max_error<max_error_threshold:
							q_breakpoints.append(q_last)
							primitives_choices.append(key)
							if key=='movec_fit':
								points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
							elif key=='movel_fit':
								points.append([curve_fit[-1]])
							else:
								points.append([q_last])
							break

					breakpoints.append(breakpoints[-1]+next_point)
					fit.append(curve_fit)
					breakpoints_out.append(breakpoints_out[-1]+len(curve_fit))


					break


			print(breakpoints)
			print(primitives_choices)
			# print(points)

		##############################check error (against fitting forward projected curve)##############################
		fit_all=np.vstack(fit)
		error=[]
		for i in range(len(fit_all)):
		    error_temp=np.linalg.norm(curve_backproj-fit_all[i],axis=1)
		    idx=np.argmin(error_temp)
		    error.append(error_temp[idx])

		error=np.array(error)
		max_error=np.max(error)
		print('max error: ', max_error)


		###plt
		###3D plot
		plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
		for i in range(len(fit)):
			ax.scatter3D(fit[i][:,0], fit[i][:,1], fit[i][:,2], c=fit[i][:,2], cmap='Greens')
		plt.show()

		return breakpoints,breakpoints_out,primitives_choices,points,fit_all




def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_ge/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###read in points backprojected
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_ge/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_backproj_x=data['X'].tolist()
	curve_backproj_y=data['Y'].tolist()
	curve_backproj_z=data['Z'].tolist()
	curve_backproj=np.vstack((curve_backproj_x, curve_backproj_y, curve_backproj_z)).T

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_ge/Curve_backproj_js.csv", names=col_names)
	curve_backproj_q1=data['q1'].tolist()
	curve_backproj_q2=data['q2'].tolist()
	curve_backproj_q3=data['q3'].tolist()
	curve_backproj_q4=data['q4'].tolist()
	curve_backproj_q5=data['q5'].tolist()
	curve_backproj_q6=data['q6'].tolist()
	curve_backproj_js=np.vstack((curve_backproj_q1, curve_backproj_q2, curve_backproj_q3,curve_backproj_q4,curve_backproj_q5,curve_backproj_q6)).T

	breakpoints,breakpoints_out,primitives_choices,points,curve_fit=fit_under_error(curve,curve_backproj,curve_backproj_js,max_error_threshold=1.)

	###insert initial configuration
	primitives_choices.insert(0,'movej_fit')

	q_all=np.array(inv(curve_fit[0],fwd(curve_backproj_js[0]).R))

	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_backproj_js[0]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_init=q_all[order[0]]


	points.insert(0,[q_init])
	print(primitives_choices)
	print(points)

	df=DataFrame({'breakpoints':breakpoints,'breakpoints_out':breakpoints_out,'primitives':primitives_choices,'points':points})
	df.to_csv('comparison/moveL+moveC/command_backproj.csv',header=True,index=False)
	df=DataFrame({'x':curve_fit[:,0],'y':curve_fit[:,1],'z':curve_fit[:,2]})
	df.to_csv('comparison/moveL+moveC/curve_fit_backproj.csv',header=True,index=False)

if __name__ == "__main__":
	main()

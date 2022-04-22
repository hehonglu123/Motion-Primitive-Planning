import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from fitting_toolbox_new import *
import sys
sys.path.append('../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from robots_def import *
from general_robotics_toolbox import *
from error_check import *
# from robotstudio_send import MotionSend

#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched self.breakpoints###############################

class greedy_fit(fitting_toolbox):
	def __init__(self,robot,curve_poly_coeff,curve_js_poly_coeff,lam_f=1758.276831, num_points=50000, orientation_weight=1):
		###robot: robot class
		###lam_f: path length
		###curve_poly_coeff: analytical xyz position 
		###curve_js_poly_coeff: analytical q
		###num_points: number of points used for greedy fitting
		###orientation_weight: weight when fitting orientation

		self.curve_poly=[]
		self.curve_js_poly=[]
		for i in range(3):
			self.curve_poly.append(np.poly1d(curve_poly_coeff[i]))
		for i in range(len(robot.joint_vel_limit)):
			self.curve_js_poly.append(np.poly1d(curve_js_poly_coeff[i]))
		self.lam=np.linspace(0,lam_f,num_points)
		self.orientation_weight=orientation_weight

		self.robot=robot

		###get curve based on lambda
		curve=np.vstack((self.curve_poly[0](self.lam),self.curve_poly[1](self.lam),self.curve_poly[2](self.lam))).T
		curve_js=np.vstack((self.curve_js_poly[0](self.lam),self.curve_js_poly[1](self.lam),self.curve_js_poly[2](self.lam),self.curve_js_poly[3](self.lam),self.curve_js_poly[4](self.lam),self.curve_js_poly[5](self.lam))).T

		super().__init__(robot,curve_js,orientation_weight,curve)
		self.slope_constraint=np.radians(180)
		self.break_early=False
		###initial primitive candidates
		self.primitives={'movel_fit':self.movel_fit_greedy,'movej_fit':self.movej_fit_greedy,'movec_fit':self.movec_fit_greedy}

	def movel_fit_greedy(self,curve,curve_js,curve_R):	###unit vector slope
		
		return self.movel_fit(curve,curve_js,curve_R,self.curve_fit[-1] if len(self.curve_fit)>0 else [],self.curve_fit_R[-1] if len(self.curve_fit_R)>0 else [])
	


	def movej_fit_greedy(self,curve,curve_js,curve_R):
		

		return self.movej_fit(curve,curve_js,curve_R,self.curve_fit_js[-1] if len(self.curve_fit_js)>0 else [])


	def movec_fit_greedy(self,curve,curve_js,curve_R):
		return self.movec_fit(curve,curve_js,curve_R,self.curve_fit[-1] if len(self.curve_fit)>0 else [],self.curve_fit_R[-1] if len(self.curve_fit_R)>0 else [])

	def fit_under_error(self,max_error_threshold,max_ori_threshold=np.radians(3)):

		step_size=int(len(self.curve)/20)
		###initialize
		self.breakpoints=[0]
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
		self.cartesian_slope_prev=[]
		self.rotation_axis_prev=[]
		self.slope_diff=[]
		self.slope_diff_ori=[]
		self.js_slope_prev=None

		while self.breakpoints[-1]<len(self.curve)-1:
			
			next_point = min(step_size,len(self.curve)-self.breakpoints[-1])
			prev_point=0
			prev_possible_point=0

			max_errors={'movel_fit':999,'movej_fit':999,'movec_fit':999}
			max_ori_errors={'movel_fit':999,'movej_fit':999,'movec_fit':999}
			###initial error map update:
			for key in self.primitives: 
				curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=self.primitives[key](self.curve[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_js[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_R[self.breakpoints[-1]:self.breakpoints[-1]+next_point])
				max_errors[key]=max_error
				max_ori_errors[key]=max_ori_error

			###bisection search self.breakpoints
			while True:
				print('index: ',self.breakpoints[-1]+next_point,'max error: ',max_errors[min(max_errors, key=max_errors.get)],'max ori error (deg): ',np.degrees(max_ori_errors[min(max_ori_errors, key=max_ori_errors.get)]))
				###bp going backward to meet threshold
				if min(list(max_errors.values()))>max_error_threshold or min(list(max_ori_errors.values()))>max_ori_threshold:
					prev_point_temp=next_point
					next_point-=int(np.abs(next_point-prev_point)/2)
					prev_point=prev_point_temp
					
					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=self.primitives[key](self.curve[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_js[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_R[self.breakpoints[-1]:self.breakpoints[-1]+next_point])
						max_errors[key]=max_error
						max_ori_errors[key]=max_ori_error



				###bp going forward to get close to threshold
				else:
					prev_possible_point=next_point
					prev_point_temp=next_point
					next_point= min(next_point + int(np.abs(next_point-prev_point)),len(self.curve)-self.breakpoints[-1])
					prev_point=prev_point_temp
					

					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=self.primitives[key](self.curve[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_js[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_R[self.breakpoints[-1]:self.breakpoints[-1]+next_point])
						max_errors[key]=max_error
						max_ori_errors[key]=max_ori_error

				# print(max_errors)
				if next_point==prev_point:
					print('stuck, restoring previous possible index')		###if ever getting stuck, restore
					next_point=max(prev_possible_point,2)
					# if self.breakpoints[-1]+next_point+1==len(self.curve)-1:
					# 	next_point=3

					primitives_added=False
					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=self.primitives[key](self.curve[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_js[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_R[self.breakpoints[-1]:self.breakpoints[-1]+next_point])
						if max_error<max_error_threshold:
							primitives_added=True
							primitives_choices.append(key)
							if key=='movec_fit':
								points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
							elif key=='movel_fit':
								points.append([curve_fit[-1]])
							else:
								points.append([curve_fit_js[-1]])
							break
					if not primitives_added:
						curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=self.movej_fit(self.curve[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_js[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_R[self.breakpoints[-1]:self.breakpoints[-1]+next_point])
						print('primitive skipped1')
						primitives_choices.append('movej_fit')
						points.append([curve_fit_js[-1]])
	
					break

				###find the closest but under max_threshold
				if min(list(max_errors.values()))<=max_error_threshold and min(list(max_ori_errors.values()))<=max_ori_threshold and np.abs(next_point-prev_point)<10:
					primitives_added=False
					for key in self.primitives: 
						curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=self.primitives[key](self.curve[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_js[self.breakpoints[-1]:self.breakpoints[-1]+next_point],self.curve_R[self.breakpoints[-1]:self.breakpoints[-1]+next_point])
						if max_error<max_error_threshold:
							primitives_added=True
							primitives_choices.append(key)
							if key=='movec_fit':
								points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
							elif key=='movel_fit':
								points.append([curve_fit[-1]])
							else:
								points.append([curve_fit_js[-1]])	
							break
					if not primitives_added:
						print('primitive skipped2')
						primitives_choices.append('movel_fit')
						points.append([curve_fit[-1]])

					break
	

			if self.break_early:
				idx=max(self.breakearly(self.curve_backproj[self.breakpoints[-1]:self.breakpoints[-1]+next_point],curve_fit),2)
			else:
				idx=next_point

			self.breakpoints.append(min(self.breakpoints[-1]+idx,len(self.curve)))
			self.curve_fit.extend(curve_fit[:len(curve_fit)-(next_point-idx)])
			self.curve_fit_R.extend(curve_fit_R[:len(curve_fit)-(next_point-idx)])

			if primitives_choices[-1]=='movej_fit':
				self.curve_fit_js.extend(curve_fit_js[:len(curve_fit)-(next_point-idx)])
			else:
				###inv here to save time
				self.curve_fit_js.extend(self.car2js(curve_fit[:len(curve_fit)-(next_point-idx)],curve_fit_R[:len(curve_fit)-(next_point-idx)]))
			###calculating ending slope
			R_diff=np.dot(curve_fit_R[0].T,curve_fit_R[-(next_point-idx)-1])
			k,theta=R2rot(R_diff)
			if len(self.cartesian_slope_prev)>0:
				self.slope_diff.append(self.get_angle(self.cartesian_slope_prev,curve_fit[1]-curve_fit[0]))
				self.slope_diff_ori.append(self.get_angle(self.rotation_axis_prev,k))

			self.rotation_axis_prev=k	
			self.cartesian_slope_prev=(self.curve_fit[-1]-self.curve_fit[-2])/np.linalg.norm(self.curve_fit[-1]-self.curve_fit[-2])
			self.js_slope_prev=(self.curve_fit_js[-1]-self.curve_fit_js[-2])/np.linalg.norm(self.curve_fit_js[-1]-self.curve_fit_js[-2])
			self.q_prev=self.curve_fit_js[-1]


			print(self.breakpoints)
			print(primitives_choices)
			
			# if len(self.breakpoints)>2:
			# 	break

		##############################check error (against fitting back projected curve)##############################

		# max_error,max_error_idx=calc_max_error(self.curve_fit,self.curve_backproj)
		# print('max error: ', max_error)
		print('slope diff: ',np.degrees(self.slope_diff))
		print('slope diff ori: ',np.degrees(self.slope_diff_ori))
		self.curve_fit=np.array(self.curve_fit)
		self.curve_fit_R=np.array(self.curve_fit_R)
		self.curve_fit_js=np.array(self.curve_fit_js)

		return self.breakpoints,primitives_choices,points



def main():
	###read in poly
	col_names=['poly_x', 'poly_y', 'poly_z','poly_direction_x', 'poly_direction_y', 'poly_direction_z'] 
	data = read_csv("../data/from_ge/Curve_in_base_frame_poly.csv", names=col_names)
	poly_x=data['poly_x'].tolist()
	poly_y=data['poly_y'].tolist()
	poly_z=data['poly_z'].tolist()
	curve_poly_coeff=np.vstack((poly_x, poly_y, poly_z))

	col_names=['poly_q1', 'poly_q2', 'poly_q3','poly_q4', 'poly_q5', 'poly_q6'] 
	data = read_csv("../data/from_ge/Curve_js_poly.csv", names=col_names)
	poly_q1=data['poly_q1'].tolist()
	poly_q2=data['poly_q2'].tolist()
	poly_q3=data['poly_q3'].tolist()
	poly_q4=data['poly_q4'].tolist()
	poly_q5=data['poly_q5'].tolist()
	poly_q6=data['poly_q6'].tolist()
	curve_js_poly_coeff=np.vstack((poly_q1, poly_q2, poly_q3,poly_q4,poly_q5,poly_q6))


	robot=abb6640(d=50)

	greedy_fit_obj=greedy_fit(robot,curve_poly_coeff,curve_js_poly_coeff, num_points=5000, orientation_weight=1)


	###set primitive choices, defaults are all 3
	# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy,'movec_fit':greedy_fit_obj.movec_fit_greedy}

	now=time.time()
	breakpoints,primitives_choices,points=greedy_fit_obj.fit_under_error(0.5)
	print('Greedy Search Time: ',time.time()-now)

	###plt
	###3D plot
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(greedy_fit_obj.curve[:,0], greedy_fit_obj.curve[:,1],greedy_fit_obj.curve[:,2], 'gray')
	
	ax.scatter3D(greedy_fit_obj.curve_fit[:,0], greedy_fit_obj.curve_fit[:,1], greedy_fit_obj.curve_fit[:,2], c=greedy_fit_obj.curve_fit[:,2], cmap='Greens')
	plt.show()

	############insert initial configuration#################
	primitives_choices.insert(0,'movej_fit')
	q_all=np.array(robot.inv(greedy_fit_obj.curve_fit[0],greedy_fit_obj.curve_fit_R[0]))
	###choose inv_kin closest to previous joints
	temp_q=q_all-greedy_fit_obj.curve_fit_js[0]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_init=q_all[order[0]]
	points.insert(0,[q_init])

	print(len(breakpoints))
	print(len(primitives_choices))
	print(len(points))

	# df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points})
	# df.to_csv('command.csv',header=True,index=False)
	# df=DataFrame({'x':greedy_fit_obj.curve_fit[:,0],'y':greedy_fit_obj.curve_fit[:,1],'z':greedy_fit_obj.curve_fit[:,2],\
	# 	'R1':greedy_fit_obj.curve_fit_R[:,0,0],'R2':greedy_fit_obj.curve_fit_R[:,0,1],'R3':greedy_fit_obj.curve_fit_R[:,0,2],\
	# 	'R4':greedy_fit_obj.curve_fit_R[:,1,0],'R5':greedy_fit_obj.curve_fit_R[:,1,1],'R6':greedy_fit_obj.curve_fit_R[:,1,2],\
	# 	'R7':greedy_fit_obj.curve_fit_R[:,2,0],'R8':greedy_fit_obj.curve_fit_R[:,2,1],'R9':greedy_fit_obj.curve_fit_R[:,2,2]})
	# df.to_csv('curve_fit.csv',header=True,index=False)
	# df=DataFrame({'j1':greedy_fit_obj.curve_fit_js[:,0],'j2':greedy_fit_obj.curve_fit_js[:,1],'j3':greedy_fit_obj.curve_fit_js[:,2],'j4':greedy_fit_obj.curve_fit_js[:,3],'j5':greedy_fit_obj.curve_fit_js[:,4],'j6':greedy_fit_obj.curve_fit_js[:,5]})
	# df.to_csv('curve_fit_js.csv',header=False,index=False)

	###check error agains original curve
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_normal_x=data['direction_x'].tolist()
	curve_normal_y=data['direction_y'].tolist()
	curve_normal_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_normal_x, curve_normal_y, curve_normal_z)).T

	# error_max=calc_max_error(greedy_fit_obj.curve_fit,curve)
	max_error,max_error_angle, max_error_idx=calc_max_error_w_normal(greedy_fit_obj.curve_fit,curve,greedy_fit_obj.curve_fit_R[:,:,-1],curve_normal)
	print(max_error,np.degrees(max_error_angle))


if __name__ == "__main__":
	main()

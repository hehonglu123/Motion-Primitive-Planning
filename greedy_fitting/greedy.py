import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from fitting_toolbox import *
import sys
sys.path.append('../toolbox')
from toolbox_circular_fit import *
from robots_def import *
from general_robotics_toolbox import *
from error_check import *
from MotionSend import *
from lambda_calc import *

#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched self.breakpoints###############################

class greedy_fit(fitting_toolbox):
	def __init__(self,robot,curve_js,min_length,max_error_threshold,max_ori_threshold=np.radians(3)):
		super().__init__(robot,curve_js[:])
		self.max_error_threshold=max_error_threshold
		self.max_ori_threshold=max_ori_threshold
		self.step=int(len(curve_js)/25)
		self.c_min_length=50
		self.min_step=int(min_length/np.average(np.diff(self.lam)))
		self.min_step_start_end=200

		self.slope_constraint=np.radians(360)
		self.dqdlam_slope=999
		self.break_early=False
		###initial primitive candidates
		self.primitives={'movel_fit':self.movel_fit_greedy,'movej_fit':self.movej_fit_greedy,'movec_fit':self.movec_fit_greedy}

	def movel_fit_greedy(self,curve,curve_js,curve_R, rl=False):	###unit vector slope
		
		return self.movel_fit(curve,curve_js,curve_R,self.curve_fit[-1] if len(self.curve_fit)>0 else [],self.curve_fit_R[-1] if len(self.curve_fit_R)>0 else [], dqdlam_prev=(self.curve_fit_js[-1]-self.curve_fit_js[-2])/(self.lam[len(self.curve_fit_js)-1]-self.lam[len(self.curve_fit_js)-2]) if len(self.curve_fit_js)>1 else [], rl=rl)
	


	def movej_fit_greedy(self,curve,curve_js,curve_R, rl=False):

		return self.movej_fit(curve,curve_js,curve_R,self.curve_fit_js[-1] if len(self.curve_fit_js)>0 else [], dqdlam_prev=(self.curve_fit_js[-1]-self.curve_fit_js[-2])/(self.lam[len(self.curve_fit_js)-1]-self.lam[len(self.curve_fit_js)-2]) if len(self.curve_fit_js)>1 else [], rl=rl)


	def movec_fit_greedy(self,curve,curve_js,curve_R, rl=False):
		return self.movec_fit(curve,curve_js,curve_R,self.curve_fit[-1] if len(self.curve_fit)>0 else [],self.curve_fit_R[-1] if len(self.curve_fit_R)>0 else [], dqdlam_prev=(self.curve_fit_js[-1]-self.curve_fit_js[-2])/(self.lam[len(self.curve_fit_js)-1]-self.lam[len(self.curve_fit_js)-2]) if len(self.curve_fit_js)>1 else [], rl=rl)

	def bisect(self,primitive,cur_idx, rl=False):

		next_point = min(self.step,len(self.curve)-self.breakpoints[-1])
		prev_point=0
		prev_possible_point=0

		while True:
			###end condition, bisection bp converges
			if next_point==prev_point:
				if rl:
					if np.max(max_error)<self.max_error_threshold and np.max(max_ori_error)<self.max_ori_threshold:
						return curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error
					else:
						next_point=max(prev_possible_point,2)
						return primitive(self.curve[cur_idx:cur_idx+next_point],self.curve_js[cur_idx:cur_idx+next_point],self.curve_R[cur_idx:cur_idx+next_point], rl=rl)
				else:
					if max_error<self.max_error_threshold and max_ori_error<self.max_ori_threshold:
						return curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error
					else:
						next_point=max(prev_possible_point,2)
						return primitive(self.curve[cur_idx:cur_idx+next_point],self.curve_js[cur_idx:cur_idx+next_point],self.curve_R[cur_idx:cur_idx+next_point], rl=rl)
			
			###end condition2, gurantee minimum segment length, excluding first and last points
			# if prev_point<self.min_step and next_point<self.min_step and self.breakpoints[-1]>0:
			# 	next_point=self.min_step
			# 	return primitive(self.curve[cur_idx:cur_idx+next_point],self.curve_js[cur_idx:cur_idx+next_point],self.curve_R[cur_idx:cur_idx+next_point], rl=rl)

			###fitting
			curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=primitive(self.curve[cur_idx:cur_idx+next_point],self.curve_js[cur_idx:cur_idx+next_point],self.curve_R[cur_idx:cur_idx+next_point], rl=rl)

			###bp going backward to meet threshold
			if rl:
				if np.max(max_error) > self.max_error_threshold or np.max(max_ori_error) > self.max_ori_threshold:
					prev_point_temp = next_point
					next_point -= int(np.abs(next_point - prev_point) / 2)
					prev_point = prev_point_temp

				###bp going forward to get close to threshold
				else:
					prev_possible_point = next_point
					prev_point_temp = next_point
					next_point = min(next_point + int(np.abs(next_point - prev_point)), len(self.curve) - cur_idx)
					prev_point = prev_point_temp
			else:
				if max_error>self.max_error_threshold or max_ori_error>self.max_ori_threshold:
					prev_point_temp=next_point
					next_point-=int(np.abs(next_point-prev_point)/2)
					prev_point=prev_point_temp

				###bp going forward to get close to threshold
				else:
					prev_possible_point=next_point
					prev_point_temp=next_point
					next_point= min(next_point + int(np.abs(next_point-prev_point)),len(self.curve)-cur_idx)
					prev_point=prev_point_temp


	def fit_under_error(self):

		###initialize
		self.breakpoints=[0]
		primitives_choices=[]
		points=[]
		q_bp=[]

		self.curve_fit=[]
		self.curve_fit_R=[]
		self.curve_fit_js=[]

		while self.breakpoints[-1]<len(self.curve)-1:

			max_errors={}
			max_ori_errors={}
			length={}
			curve_fit={}
			curve_fit_R={}
			curve_fit_js={}

			###bisection search for each primitive 
			for key in self.primitives: 
				curve_fit[key],curve_fit_R[key],curve_fit_js[key],max_errors[key],max_ori_errors[key]=self.bisect(self.primitives[key],self.breakpoints[-1])
				length[key]=len(curve_fit[key])
			###find best primitive
			if length['movec_fit']==length['movel_fit'] and length['movel_fit']==length['movej_fit']:
				key=min(max_errors, key=max_errors.get)
			else:
				key=max(length, key=length.get)

			###moveC length thresholding (>50mm)
			if key=='movec_fit' and np.linalg.norm(curve_fit['movec_fit'][-1]-curve_fit['movec_fit'][0])<self.c_min_length:
				key='movel_fit'


			
			primitives_choices.append(key)
			self.breakpoints.append(min(self.breakpoints[-1]+len(curve_fit[key]),len(self.curve)))
			self.curve_fit.extend(curve_fit[key])
			self.curve_fit_R.extend(curve_fit_R[key])

			if key=='movej_fit':
				self.curve_fit_js.extend(curve_fit_js[key])
			else:
				###inv here to save time
				if len(self.curve_fit_js)>1:
					q_init=self.curve_fit_js[-1]
				else:
					q_init=self.curve_js[0]

				curve_fit_js=car2js(self.robot,q_init,curve_fit[key],curve_fit_R[key])
			
				self.curve_fit_js.extend(curve_fit_js)

			if key=='movec_fit':
				points.append([curve_fit[key][int(len(curve_fit[key])/2)],curve_fit[key][-1]])
				q_bp.append([curve_fit_js[int(len(curve_fit_R[key])/2)],curve_fit_js[-1]])
			elif key=='movel_fit':
				points.append([curve_fit[key][-1]])
				q_bp.append([curve_fit_js[-1]])
			else:
				points.append([curve_fit[key][-1]])
				q_bp.append([curve_fit_js[key][-1]])


			print(self.breakpoints)
			print(primitives_choices)
			print(max_errors[key],max_ori_errors[key])
			

		##############################check error (against fitting back projected curve)##############################

		# max_error,max_error_idx=calc_max_error(self.curve_fit,self.curve_backproj)
		# print('max error: ', max_error)

		self.curve_fit=np.array(self.curve_fit)
		self.curve_fit_R=np.array(self.curve_fit_R)
		self.curve_fit_js=np.array(self.curve_fit_js)

		return self.breakpoints,primitives_choices,points,q_bp

	def merge_bp(self,breakpoints,primitives_choices,points,q_bp):
		###merge closely programmed bp's
		bp_diff=np.diff(breakpoints)
		close_indices=np.argwhere(bp_diff<self.min_step).flatten().astype(int)
		#remove first and last bp if there
		if 0 in close_indices and breakpoints[1]>self.min_step_start_end:
			close_indices=close_indices[1:]
		if len(breakpoints)-2 in close_indices:
			close_indices=close_indices[:-1]

		###merge closely programmed points
		# fit_primitives={'movel_fit':movel_fit,'movec_fit':movec_fit,'movej_fit':movej_fit}
		# for idx in close_indices:
		# 	new_bp=int((breakpoints[idx]+breakpoints[idx+1])/2)
		# 	if primitives_choices=='movej_fit':
		# 		curve_fit,curve_fit_R,curve_fit_js,_,_=fit_primitives[primitives_choices[idx]](curve[breakpoints[idx-1]:new_bp],curve_js[breakpoints[idx-1]:new_bp],curve_R[breakpoints[idx-1]:new_bp],p_constraint=curve_fit_js[breakpoints[idx-1]])
		# 	else:
		# 		curve_fit,curve_fit_R,_,_,_=fit_primitives[primitives_choices[idx]](curve[breakpoints[idx-1]:new_bp],curve_js[breakpoints[idx-1]:new_bp],curve_R[breakpoints[idx-1]:new_bp],p_constraint=curve_fit[breakpoints[idx-1]],R_constraint=self.robot.fwd(curve_fit_js[breakpoints[idx-1]]).R)
		# 		curve_fit_js=car2js(self.robot,curve_fit_js[breakpoints[idx-1]],curve_fit,curve_fit_R)
		# 	points[idx][-1]=curve_fit[-1]
		# 	q_bp[idx][-1]=curve_fit_js[-1]

		###remove old breakpoints
		indicies2remove=(close_indices+1).tolist()
		indicies2remove.reverse()
		for i in indicies2remove:
			del breakpoints[i]
			del primitives_choices[i]
			del points[i]
			del q_bp[i]

		#second last point removal
		if breakpoints[-1]-breakpoints[-2]<self.min_step_start_end:
			points[-2][-1]=points[-1][-1]
			q_bp[-2][-1]=q_bp[-1][-1]
			breakpoints[-2]=breakpoints[-1]

			del breakpoints[-1]
			del primitives_choices[-1]
			del points[-1]
			del q_bp[-1]



		return breakpoints,primitives_choices,points,q_bp

def main():
	dataset='from_NX/'
	solution_dir='curve_pose_opt2_R/'
	data_dir="../data/"+dataset+solution_dir

	###read in points
	# curve_js = read_csv("../train_data/wood/Curve_js.csv",header=None).values
	curve_js = read_csv(data_dir+'Curve_js.csv',header=None).values

	robot=abb6640(d=50)

	max_error_threshold=0.02
	min_length=50
	greedy_fit_obj=greedy_fit(robot,curve_js, min_length=min_length,max_error_threshold=max_error_threshold)

	###set primitive choices, defaults are all 3
	# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy,'movec_fit':greedy_fit_obj.movec_fit_greedy}

	# greedy_fit_obj.primitives={'movej_fit':greedy_fit_obj.movej_fit_greedy}
	# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy}
	# greedy_fit_obj.primitives={'movec_fit':greedy_fit_obj.movec_fit_greedy}

	# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy,'movej_fit':greedy_fit_obj.movej_fit_greedy}

	breakpoints,primitives_choices,points,q_bp=greedy_fit_obj.fit_under_error()
	print('slope diff js (deg): ', greedy_fit_obj.get_slope_js(greedy_fit_obj.curve_fit_js,breakpoints))
	
	############insert initial configuration#################
	primitives_choices.insert(0,'moveabsj_fit')
	points.insert(0,[greedy_fit_obj.curve_fit[0]])
	q_bp.insert(0,[greedy_fit_obj.curve_fit_js[0]])


	breakpoints,primitives_choices,points,q_bp=greedy_fit_obj.merge_bp(breakpoints,primitives_choices,points,q_bp)
	# print(breakpoints)
	###plt
	###3D plot
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(greedy_fit_obj.curve[:,0], greedy_fit_obj.curve[:,1],greedy_fit_obj.curve[:,2], 'gray',label='original')
	
	ax.plot3D(greedy_fit_obj.curve_fit[:,0], greedy_fit_obj.curve_fit[:,1], greedy_fit_obj.curve_fit[:,2],'green',label='fitting')
	plt.legend()
	plt.show()

	

	print(len(breakpoints))
	print(len(primitives_choices))
	print(len(points))
	print(len(q_bp))

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points,'q_bp':q_bp})
	df.to_csv('greedy_output/command.csv',header=True,index=False)
	df=DataFrame({'x':greedy_fit_obj.curve_fit[:,0],'y':greedy_fit_obj.curve_fit[:,1],'z':greedy_fit_obj.curve_fit[:,2],\
		'R1':greedy_fit_obj.curve_fit_R[:,0,0],'R2':greedy_fit_obj.curve_fit_R[:,0,1],'R3':greedy_fit_obj.curve_fit_R[:,0,2],\
		'R4':greedy_fit_obj.curve_fit_R[:,1,0],'R5':greedy_fit_obj.curve_fit_R[:,1,1],'R6':greedy_fit_obj.curve_fit_R[:,1,2],\
		'R7':greedy_fit_obj.curve_fit_R[:,2,0],'R8':greedy_fit_obj.curve_fit_R[:,2,1],'R9':greedy_fit_obj.curve_fit_R[:,2,2]})
	df.to_csv('greedy_output/curve_fit.csv',header=True,index=False)
	DataFrame(greedy_fit_obj.curve_fit_js).to_csv('greedy_output/curve_fit_js.csv',header=False,index=False)

def greedy_execute():
	ms = MotionSend()
	###read in points
	# curve_js=read_csv("../train_data/from_NX/Curve_js.csv",header=None).values
	curve_js = read_csv("../data/wood/Curve_js.csv", header=None).values

	robot=abb6640(d=50)

	greedy_fit_obj=greedy_fit(robot,curve_js[::10],0.5)
	# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy}
	greedy_fit_obj.primitives={'movej_fit':greedy_fit_obj.movej_fit_greedy}
	# greedy_fit_obj.primitives={'movec_fit':greedy_fit_obj.movec_fit_greedy}

	###greedy fitting
	breakpoints,primitives_choices,points, q_bp=greedy_fit_obj.fit_under_error()

	############insert initial configuration#################
	primitives_choices.insert(0,'movej_fit')
	points.insert(0,[greedy_fit_obj.curve_fit[0]])
	q_bp.insert(0,[greedy_fit_obj.curve_fit_js[0]])

	###extension
	points,q_bp=ms.extend(robot,q_bp,primitives_choices,breakpoints,points)

	#######################RS execution################################
	from io import StringIO
	
	logged_data=ms.exec_motions(robot,primitives_choices,breakpoints,points,q_bp,v500,z10)
	StringData=StringIO(logged_data)
	df = read_csv(StringData, sep =",")
	##############################train_data analysis#####################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=False)
	#############################chop extension off##################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,greedy_fit_obj.curve[0,:3],greedy_fit_obj.curve[-1,:3])
	error,angle_error=calc_all_error_w_normal(curve_exe,greedy_fit_obj.curve,curve_exe_R[:,:,-1],greedy_fit_obj.curve_R[:,:,-1])

	print('time: ',timestamp[-1]-timestamp[0],'error: ',np.max(error),'normal error: ',np.max(angle_error))

if __name__ == "__main__":
	# greedy_execute()
	main()

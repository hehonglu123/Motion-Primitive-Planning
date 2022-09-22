import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from fitting_toolbox_dual import *
import sys, yaml

from toolbox_circular_fit import *
from robots_def import *
from general_robotics_toolbox import *
from error_check import *
from MotionSend import *
from dual_arm import *
#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched self.breakpoints###############################

class greedy_fit(fitting_toolbox):
	def __init__(self,robot1,robot2,curve_js1,curve_js2,base2_p,base2_R,min_length,max_error_threshold,max_ori_threshold=np.radians(3)):
		super().__init__(robot1,robot2,curve_js1,curve_js2,base2_p,base2_R)
		self.max_error_threshold=max_error_threshold
		self.max_ori_threshold=max_ori_threshold
		self.step=int(len(curve_js1)/25)

		self.slope_constraint=np.radians(180)
		self.break_early=False
		###initial primitive candidates
		self.primitives={'movel_fit':self.movel_fit,'movej_fit':self.movej_fit,'movec_fit':self.movec_fit}

	def update_dict(self,curve1,curve2,curve_js1,curve_js2,curve_R1,curve_R2,curve_relative,curve_relative_R):
		###form new error dict
		error_dict={}
		ori_error_dict={}
		curve_fit_dict1={}
		curve_fit_R_dict1={}
		curve_fit_dict2={}
		curve_fit_R_dict2={}

		###fit all 3 for all robots first
		for key1 in self.primitives:
			if 'movej' in key1:
				curve_fit_dict1[key1],curve_fit_R_dict1[key1],_,_,_=self.primitives[key1](curve1,curve_js1,curve_R1,self.robot1,self.curve_fit_js1[-1] if len(self.curve_fit_js1)>0 else [])
			else:
				curve_fit_dict1[key1],curve_fit_R_dict1[key1],_,_,_=self.primitives[key1](curve1,curve_js1,curve_R1,self.robot1,self.curve_fit1[-1] if len(self.curve_fit1)>0 else [],self.curve_fit_R1[-1] if len(self.curve_fit_R1)>0 else [])
		for key2 in self.primitives:
			if 'movej' in key2:
				curve_fit_dict2[key2],curve_fit_R_dict2[key2],_,_,_=self.primitives[key2](curve2,curve_js2,curve_R2,self.robot2,self.curve_fit_js2[-1] if len(self.curve_fit_js2)>0 else [])
			else:
				curve_fit_dict2[key2],curve_fit_R_dict2[key2],_,_,_=self.primitives[key2](curve2,curve_js2,curve_R2,self.robot2,self.curve_fit2[-1] if len(self.curve_fit2)>0 else [],self.curve_fit_R2[-1] if len(self.curve_fit_R2)>0 else [])

		###update relative error
		for key1 in self.primitives:
			for key2 in self.primitives:
				relative_path_fit=[]
				relative_norm_fit=[]
				ori_error=[]
				for i in range(len(curve1)):
					#convert to robot2 tool frame
					pose2_world_now_R=self.base2_R@curve_fit_R_dict2[key2][i]
					pose2_world_now_p=self.base2_R@curve_fit_dict2[key2][i]+self.base2_p

					relative_path_fit.append(pose2_world_now_R.T@(curve_fit_dict1[key1][i]-pose2_world_now_p))
					relative_norm_fit.append(pose2_world_now_R.T@curve_fit_R_dict1[key1][i][:,-1])

					ori_error.append(get_angle(curve_relative_R[i][:,-1],relative_norm_fit[-1]))


				error_dict[(key1,key2)]=np.max(np.linalg.norm(curve_relative-np.array(relative_path_fit),axis=1))
				ori_error_dict[(key1,key2)]=np.max(ori_error)

		return error_dict,ori_error_dict,curve_fit_dict1,curve_fit_R_dict1,curve_fit_dict2,curve_fit_R_dict2

	def bisect(self,cur_idx):

		next_point = min(self.step,len(self.curve_js1)-self.breakpoints[-1])
		prev_point=0
		prev_possible_point=0


		while True:
			###end condition
			if next_point==prev_point:
				###TODO: may not be the same comb with min value
				if min(error_dict.values())<self.max_error_threshold and min(ori_error_dict.values())<self.max_ori_threshold:
					##find min comb
					primitive_comb=min(error_dict, key=error_dict.get)
					print('min relative error: ',min(error_dict.values()))
					return primitive_comb[0],primitive_comb[1],curve_fit_dict1[primitive_comb[0]],curve_fit_dict2[primitive_comb[1]],curve_fit_R_dict1[primitive_comb[0]],curve_fit_R_dict2[primitive_comb[1]]

				else:
					next_point=max(prev_possible_point,2)
					indices=range(cur_idx,cur_idx+next_point)
					error_dict,ori_error_dict,curve_fit_dict1,curve_fit_R_dict1,curve_fit_dict2,curve_fit_R_dict2=\
						self.update_dict(self.curve1[indices],self.curve2[indices],self.curve_js1[indices],self.curve_js2[indices],self.curve_R1[indices],self.curve_R2[indices],self.relative_path[indices],self.relative_R[indices])
					##find min comb
					primitive_comb=min(error_dict, key=error_dict.get)
					print('min relative error: ',min(error_dict.values()))
					return primitive_comb[0],primitive_comb[1],curve_fit_dict1[primitive_comb[0]],curve_fit_dict2[primitive_comb[1]],curve_fit_R_dict1[primitive_comb[0]],curve_fit_R_dict2[primitive_comb[1]]

			###fitting
			indices=range(cur_idx,cur_idx+next_point)
			error_dict,ori_error_dict,curve_fit_dict1,curve_fit_R_dict1,curve_fit_dict2,curve_fit_R_dict2=\
				self.update_dict(self.curve1[indices],self.curve2[indices],self.curve_js1[indices],self.curve_js2[indices],self.curve_R1[indices],self.curve_R2[indices],self.relative_path[indices],self.relative_R[indices])

			###bp going backward to meet threshold
			if min(error_dict.values())>self.max_error_threshold or min(ori_error_dict.values())>self.max_ori_threshold:
				prev_point_temp=next_point
				next_point-=int(np.abs(next_point-prev_point)/2)
				prev_point=prev_point_temp

			###bp going forward to get close to threshold
			else:
				prev_possible_point=next_point
				prev_point_temp=next_point
				next_point= min(next_point + int(np.abs(next_point-prev_point)),len(self.curve_js1)-cur_idx)
				prev_point=prev_point_temp


	def fit_under_error(self):

		###initialize
		self.breakpoints=[0]
		primitives_choices1=[]
		points1=[]
		q_bp1=[]
		primitives_choices2=[]
		points2=[]
		q_bp2=[]

		self.curve_fit1=[]
		self.curve_fit_R1=[]
		self.curve_fit_js1=[]
		self.curve_fit2=[]
		self.curve_fit_R2=[]
		self.curve_fit_js2=[]

		while self.breakpoints[-1]<len(self.relative_path)-1:
			
			# max_errors1={}
			# max_ori_errors1={}
			# length1={}
			# curve_fit1={}
			# curve_fit_R1={}
			# curve_fit_js1={}

			###bisection search for each primitive 
			###TODO: pass curve_js from j fit
			primitive1,primitive2,curve_fit1,curve_fit2,curve_fit_R1,curve_fit_R2=self.bisect(self.breakpoints[-1])

			###solve inv_kin here
			if len(self.curve_fit_js1)>1:
				self.curve_fit_js1.extend(car2js(self.robot1,self.curve_fit_js1[-1],curve_fit1,curve_fit_R1))
				self.curve_fit_js2.extend(car2js(self.robot2,self.curve_fit_js2[-1],curve_fit2,curve_fit_R2))
			else:
				self.curve_fit_js1.extend(car2js(self.robot1,self.curve_js1[0],curve_fit1,curve_fit_R1))
				self.curve_fit_js2.extend(car2js(self.robot2,self.curve_js2[0],curve_fit2,curve_fit_R2))

			###generate output
			if primitive1=='movec_fit':
				points1.append([curve_fit1[int(len(curve_fit1)/2)],curve_fit1[-1]])
				q_bp1.append([self.curve_fit_js1[int(len(curve_fit_R1)/2)],self.curve_fit_js1[-1]])
			elif primitive1=='movel_fit':
				points1.append([curve_fit1[-1]])
				q_bp1.append([self.curve_fit_js1[-1]])
			else:
				points1.append([curve_fit1[-1]])
				q_bp1.append([self.curve_fit_js1[-1]])

			if primitive2=='movec_fit':
				points2.append([curve_fit2[int(len(curve_fit2)/2)],curve_fit2[-1]])
				q_bp2.append([self.curve_fit_js2[int(len(curve_fit_R2)/2)],self.curve_fit_js2[-1]])
			elif primitive2=='movel_fit':
				points2.append([curve_fit2[-1]])
				q_bp2.append([self.curve_fit_js2[-1]])
			else:
				points2.append([curve_fit2[-1]])
				q_bp2.append([self.curve_fit_js2[-1]])
			
			primitives_choices1.append(primitive1)
			primitives_choices2.append(primitive2)

			self.breakpoints.append(min(self.breakpoints[-1]+len(curve_fit1),len(self.curve1)))
			self.curve_fit1.extend(curve_fit1)
			self.curve_fit_R1.extend(curve_fit_R1)
			self.curve_fit2.extend(curve_fit2)
			self.curve_fit_R2.extend(curve_fit_R2)



			print(self.breakpoints)
			print(primitives_choices1)
			

		##############################check error (against fitting back projected curve)##############################

		# max_error,max_error_idx=calc_max_error(self.curve_fit,self.curve_backproj)
		# print('max error: ', max_error)

		self.curve_fit1=np.array(self.curve_fit1)
		self.curve_fit_R1=np.array(self.curve_fit_R1)
		self.curve_fit_js1=np.array(self.curve_fit_js1)

		self.curve_fit2=np.array(self.curve_fit2)
		self.curve_fit_R2=np.array(self.curve_fit_R2)
		self.curve_fit_js2=np.array(self.curve_fit_js2)

		self.curve_fit2_global=(self.base2_R@self.curve_fit2.T).T+np.tile(self.base2_p,(len(self.curve_fit2),1))



		return np.array(self.breakpoints),primitives_choices1,points1,q_bp1,primitives_choices2,points2,q_bp2

	def merge_bp(self,breakpoints,primitives_choices1,points1,q_bp1,primitives_choices2,points2,q_bp2):
		points1_np=np.array([item[0] for item in points1])
		points2_np=np.array([item[0] for item in points2])
		


def main():
	###read in points
	dataset='from_NX/'
	data_dir="../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo3/'
	
	relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)

	min_length=20
	greedy_fit_obj=greedy_fit(robot1,robot2,curve_js1[::1],curve_js2[::1],base2_p,base2_R,min_length,0.2)


	###set primitive choices, defaults are all 3
	greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit,'movec_fit':greedy_fit_obj.movec_fit}

	breakpoints,primitives_choices1,points1,q_bp1,primitives_choices2,points2,q_bp2=greedy_fit_obj.fit_under_error()

	###plt
	###3D plot in global frame
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(greedy_fit_obj.curve_fit1[:,0], greedy_fit_obj.curve_fit1[:,1],greedy_fit_obj.curve_fit1[:,2], 'gray', label='arm1')
	ax.plot3D(greedy_fit_obj.curve_fit2_global[:,0], greedy_fit_obj.curve_fit2_global[:,1],greedy_fit_obj.curve_fit2_global[:,2], 'green', label='arm2')
	plt.legend()

	###3D plot in robot2 tool frame
	relative_path_fit=[]
	for i in range(len(greedy_fit_obj.curve_fit1)):
		#convert to robot2 tool frame
		pose2_world_now_R=greedy_fit_obj.base2_R@greedy_fit_obj.curve_fit_R2[i]
		pose2_world_now_p=greedy_fit_obj.base2_R@greedy_fit_obj.curve_fit2[i]+greedy_fit_obj.base2_p
		relative_path_fit.append(pose2_world_now_R.T@(greedy_fit_obj.curve_fit1[i]-pose2_world_now_p))

	relative_path_fit=np.array(relative_path_fit)
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(greedy_fit_obj.relative_path[:,0], greedy_fit_obj.relative_path[:,1],greedy_fit_obj.relative_path[:,2], 'gray', label='original')
	ax.plot3D(relative_path_fit[:,0], relative_path_fit[:,1],relative_path_fit[:,2], 'green', label='fitting')
	plt.legend()
	plt.show()

	############insert initial configuration#################
	primitives_choices1.insert(0,'moveabsj_fit')
	points1.insert(0,[greedy_fit_obj.curve_fit1[0]])
	q_bp1.insert(0,[greedy_fit_obj.curve_fit_js1[0]])

	primitives_choices2.insert(0,'moveabsj_fit')
	points2.insert(0,[greedy_fit_obj.curve_fit2[0]])
	q_bp2.insert(0,[greedy_fit_obj.curve_fit_js2[0]])

	print(len(breakpoints))
	print(len(primitives_choices1))
	print(len(points1))

	###shift breakpoints
	breakpoints[1:]=breakpoints[1:]-1

	###save arm1
	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices1,'p_bp':points1,'q_bp':q_bp1})
	df.to_csv('greedy_dual_output/command1.csv',header=True,index=False)
	df=DataFrame({'x':greedy_fit_obj.curve_fit1[:,0],'y':greedy_fit_obj.curve_fit1[:,1],'z':greedy_fit_obj.curve_fit1[:,2],\
		'R1':greedy_fit_obj.curve_fit_R1[:,0,0],'R2':greedy_fit_obj.curve_fit_R1[:,0,1],'R3':greedy_fit_obj.curve_fit_R1[:,0,2],\
		'R4':greedy_fit_obj.curve_fit_R1[:,1,0],'R5':greedy_fit_obj.curve_fit_R1[:,1,1],'R6':greedy_fit_obj.curve_fit_R1[:,1,2],\
		'R7':greedy_fit_obj.curve_fit_R1[:,2,0],'R8':greedy_fit_obj.curve_fit_R1[:,2,1],'R9':greedy_fit_obj.curve_fit_R1[:,2,2]})
	df.to_csv('greedy_dual_output/curve_fit1.csv',header=True,index=False)
	DataFrame(greedy_fit_obj.curve_fit_js1).to_csv('greedy_dual_output/curve_fit_js1.csv',header=False,index=False)

	###save arm2
	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices2,'p_bp':points2,'q_bp':q_bp2})
	df.to_csv('greedy_dual_output/command2.csv',header=True,index=False)
	df=DataFrame({'x':greedy_fit_obj.curve_fit2[:,0],'y':greedy_fit_obj.curve_fit2[:,1],'z':greedy_fit_obj.curve_fit2[:,2],\
		'R1':greedy_fit_obj.curve_fit_R2[:,0,0],'R2':greedy_fit_obj.curve_fit_R2[:,0,1],'R3':greedy_fit_obj.curve_fit_R2[:,0,2],\
		'R4':greedy_fit_obj.curve_fit_R2[:,1,0],'R5':greedy_fit_obj.curve_fit_R2[:,1,1],'R6':greedy_fit_obj.curve_fit_R2[:,1,2],\
		'R7':greedy_fit_obj.curve_fit_R2[:,2,0],'R8':greedy_fit_obj.curve_fit_R2[:,2,1],'R9':greedy_fit_obj.curve_fit_R2[:,2,2]})
	df.to_csv('greedy_dual_output/curve_fit2.csv',header=True,index=False)
	DataFrame(greedy_fit_obj.curve_fit_js2).to_csv('greedy_dual_output/curve_fit_js2.csv',header=False,index=False)


if __name__ == "__main__":
	main()

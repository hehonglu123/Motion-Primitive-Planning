import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from fitting_toolbox_dual import *
import sys, yaml
sys.path.append('../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from robots_def import *
from general_robotics_toolbox import *
from error_check import *
from MotionSend import *

#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched self.breakpoints###############################

class greedy_fit(fitting_toolbox):
	def __init__(self,robot1,robot2,curve_js1,curve_js2,base2_p,base2_R,max_error_threshold,max_ori_threshold=np.radians(3)):
		super().__init__(robot1,robot2,curve_js1,curve_js2,base2_p,base2_R)
		self.max_error_threshold=max_error_threshold
		self.max_ori_threshold=max_ori_threshold
		self.step=int(len(curve_js1)/25)

		self.slope_constraint=np.radians(180)
		self.break_early=False
		###initial primitive candidates
		self.primitives={'movel_fit':self.movel_fit_greedy,'movej_fit':self.movej_fit_greedy,'movec_fit':self.movec_fit_greedy}

	def movel_fit_greedy(self,curve,curve_js,curve_R,robot_idx):	###unit vector slope
		if robot_idx==1:
			return self.movel_fit(curve,curve_js,curve_R,robot_idx,self.curve_fit1[-1] if len(self.curve_fit1)>0 else [],self.curve_fit_R1[-1] if len(self.curve_fit_R1)>0 else [])
		if robot_idx==2:
			return self.movel_fit(curve,curve_js,curve_R,robot_idx,self.curve_fit2[-1] if len(self.curve_fit2)>0 else [],self.curve_fit_R2[-1] if len(self.curve_fit_R2)>0 else [])
	


	def movej_fit_greedy(self,curve,curve_js,curve_R,robot_idx):
		if robot_idx==1:
			return self.movej_fit(curve,curve_js,curve_R,robot_idx,self.curve_fit_js1[-1] if len(self.curve_fit_js1)>0 else [])
		if robot_idx==2:
			return self.movej_fit(curve,curve_js,curve_R,robot_idx,self.curve_fit_js2[-1] if len(self.curve_fit_js2)>0 else [])


	def movec_fit_greedy(self,curve,curve_js,curve_R,robot_idx):
		if robot_idx==1:
			return self.movec_fit(curve,curve_js,curve_R,robot_idx,self.curve_fit1[-1] if len(self.curve_fit1)>0 else [],self.curve_fit_R1[-1] if len(self.curve_fit_R1)>0 else [])
		if robot_idx==2:
			return self.movec_fit(curve,curve_js,curve_R,robot_idx,self.curve_fit2[-1] if len(self.curve_fit2)>0 else [],self.curve_fit_R2[-1] if len(self.curve_fit_R2)>0 else [])
	def bisect(self,primitive,cur_idx,curve,curve_js,curve_R,robot_idx):
		next_point = min(self.step,len(curve)-self.breakpoints[-1])
		prev_point=0
		prev_possible_point=0

		while True:
			###end condition
			if next_point==prev_point:
				if max_error<self.max_error_threshold and max_ori_error<self.max_ori_threshold:
					return curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error
				else:
					next_point=max(prev_possible_point,2)
					return primitive(curve[cur_idx:cur_idx+next_point],curve_js[cur_idx:cur_idx+next_point],curve_R[cur_idx:cur_idx+next_point],robot_idx)
			###fitting
			curve_fit,curve_fit_R,curve_fit_js,max_error,max_ori_error=primitive(curve[cur_idx:cur_idx+next_point],curve_js[cur_idx:cur_idx+next_point],curve_R[cur_idx:cur_idx+next_point],robot_idx)

			###bp going backward to meet threshold
			if max_error>self.max_error_threshold or max_ori_error>self.max_ori_threshold:
				prev_point_temp=next_point
				next_point-=int(np.abs(next_point-prev_point)/2)
				prev_point=prev_point_temp

			###bp going forward to get close to threshold
			else:
				prev_possible_point=next_point
				prev_point_temp=next_point
				next_point= min(next_point + int(np.abs(next_point-prev_point)),len(curve)-cur_idx)
				prev_point=prev_point_temp


	def fit_under_error(self):

		###initialize
		self.breakpoints=[0]
		primitives_choices1=[]
		points1=[]
		primitives_choices2=[]
		points2=[]

		self.curve_fit1=[]
		self.curve_fit_R1=[]
		self.curve_fit_js1=[]
		self.curve_fit2=[]
		self.curve_fit_R2=[]
		self.curve_fit_js2=[]

		while self.breakpoints[-1]<len(self.relative_path)-1:
			
			max_errors1={}
			max_ori_errors1={}
			length1={}
			curve_fit1={}
			curve_fit_R1={}
			curve_fit_js1={}

			###bisection search for each primitive 
			for key in self.primitives: 
				curve_fit1[key],curve_fit_R1[key],curve_fit_js1[key],max_errors1[key],max_ori_errors1[key]=self.bisect(self.primitives[key],self.breakpoints[-1],self.curve1,self.curve_js1,self.curve_R1,1)
				length1[key]=len(curve_fit1[key])
			###find best primitive
			key1=max(length1, key=length1.get)

			max_errors2={}
			max_ori_errors2={}
			length2={}
			curve_fit2={}
			curve_fit_R2={}
			curve_fit_js2={}

			###bisection search for each primitive 
			for key in self.primitives: 
				curve_fit2[key],curve_fit_R2[key],curve_fit_js2[key],max_errors2[key],max_ori_errors2[key]=self.bisect(self.primitives[key],self.breakpoints[-1],self.curve2,self.curve_js2,self.curve_R2,2)
				length2[key]=len(curve_fit2[key])
			###find best primitive
			key2=max(length2, key=length2.get)

			print(max_errors1,max_ori_errors1)

			if length1[key1]>length2[key2]:
				curve_fit1[key1],curve_fit_R1[key1],curve_fit_js1[key1],max_errors1[key1],max_ori_errors1[key1]=\
					self.primitives[key1](self.curve1[self.breakpoints[-1]:self.breakpoints[-1]+length2[key2]],self.curve_js1[self.breakpoints[-1]:self.breakpoints[-1]+length2[key2]],self.curve_R1[self.breakpoints[-1]:self.breakpoints[-1]+length2[key2]],1)
			else:
				curve_fit2[key2],curve_fit_R2[key2],curve_fit_js2[key2],max_errors2[key2],max_ori_errors2[key2]=\
					self.primitives[key2](self.curve2[self.breakpoints[-1]:self.breakpoints[-1]+length1[key1]],self.curve_js2[self.breakpoints[-1]:self.breakpoints[-1]+length1[key1]],self.curve_R2[self.breakpoints[-1]:self.breakpoints[-1]+length1[key1]],2)



			if key1=='movec_fit':
				points1.append([curve_fit1[key1][int(len(curve_fit1[key1])/2)],curve_fit1[key1][-1]])
			elif key1=='movel_fit':
				points1.append([curve_fit1[key1][-1]])
			else:
				points1.append([curve_fit_js1[key1][-1]])

			if key2=='movec_fit':
				points2.append([curve_fit2[key2][int(len(curve_fit2[key2])/2)],curve_fit[key2][-1]])
			elif key2=='movel_fit':
				points2.append([curve_fit2[key2][-1]])
			else:
				points2.append([curve_fit_js2[key2][-1]])
			
			primitives_choices1.append(key1)
			primitives_choices2.append(key2)

			self.breakpoints.append(min(self.breakpoints[-1]+len(curve_fit1[key1]),len(self.curve1)))
			self.curve_fit1.extend(curve_fit1[key1])
			self.curve_fit_R1.extend(curve_fit_R1[key1])
			self.curve_fit2.extend(curve_fit2[key2])
			self.curve_fit_R2.extend(curve_fit_R2[key2])

			if key1=='movej_fit':
				self.curve_fit_js1.extend(curve_fit_js1[key1])
			else:
				###inv here to save time
				if len(self.curve_fit_js1)>1:
					self.curve_fit_js1.extend(self.car2js(self.robot1,curve_fit1[key1],curve_fit_R1[key1],self.curve_fit_js1[-1]))
				else:
					self.curve_fit_js1.extend(self.car2js(self.robot1,curve_fit1[key1],curve_fit_R1[key1],self.curve_js1[0]))

			if key2=='movej_fit':
				self.curve_fit_js2.extend(curve_fit_js2[key2])
			else:
				###inv here to save time
				if len(self.curve_fit_js2)>1:
					self.curve_fit_js2.extend(self.car2js(self.robot2,curve_fit2[key2],curve_fit_R2[key2],self.curve_fit_js2[-1]))
				else:
					self.curve_fit_js2.extend(self.car2js(self.robot2,curve_fit2[key2],curve_fit_R2[key2],self.curve_js2[0]))


			# print(self.breakpoints)
			# print(primitives_choices1)
			# print(max_errors1[key1],max_ori_errors1[key1])
			

		##############################check error (against fitting back projected curve)##############################

		# max_error,max_error_idx=calc_max_error(self.curve_fit,self.curve_backproj)
		# print('max error: ', max_error)

		self.curve_fit1=np.array(self.curve_fit1)
		self.curve_fit_R1=np.array(self.curve_fit_R1)
		self.curve_fit_js1=np.array(self.curve_fit_js1)

		self.curve_fit2=np.array(self.curve_fit2)
		self.curve_fit_R2=np.array(self.curve_fit_R2)
		self.curve_fit_js2=np.array(self.curve_fit_js2)

		return self.breakpoints,primitives_choices1,points1,primitives_choices2,points2



def main():
	###read in points
	curve_js1 = read_csv("../constraint_solver/dual_arm/trajectory/arm1.csv",header=None).values
	curve_js2 = read_csv("../constraint_solver/dual_arm/trajectory/arm2.csv",header=None).values
	###define robots
	robot1=abb1200(d=50)
	robot2=abb6640()
	###read in robot2 pose
	with open('../constraint_solver/dual_arm/trajectory/abb6640.yaml') as file:
		H_6640 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	greedy_fit_obj=greedy_fit(robot1,robot2,curve_js1,curve_js2,H_6640[:-1,-1],H_6640[:-1,:-1],0.5)


	###set primitive choices, defaults are all 3
	greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy,'movec_fit':greedy_fit_obj.movec_fit_greedy}

	breakpoints,primitives_choices1,points1,primitives_choices2,points2=greedy_fit_obj.fit_under_error()

	###plt
	###3D plot
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(greedy_fit_obj.curve[:,0], greedy_fit_obj.curve[:,1],greedy_fit_obj.curve[:,2], 'gray')
	
	ax.scatter3D(greedy_fit_obj.curve_fit[:,0], greedy_fit_obj.curve_fit[:,1], greedy_fit_obj.curve_fit[:,2], c=greedy_fit_obj.curve_fit[:,2], cmap='Greens')
	plt.show()

	############insert initial configuration#################
	# primitives_choices.insert(0,'movej_fit')
	# q_all=np.array(robot.inv(greedy_fit_obj.curve_fit[0],greedy_fit_obj.curve_fit_R[0]))
	# ###choose inv_kin closest to previous joints
	# temp_q=q_all-curve_js[0]
	# order=np.argsort(np.linalg.norm(temp_q,axis=1))
	# q_init=q_all[order[0]]
	# points.insert(0,[q_init])

	# print(len(breakpoints))
	# print(len(primitives_choices))
	# print(len(points))

	# df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points})
	# df.to_csv('command.csv',header=True,index=False)
	# df=DataFrame({'x':greedy_fit_obj.curve_fit[:,0],'y':greedy_fit_obj.curve_fit[:,1],'z':greedy_fit_obj.curve_fit[:,2],\
	# 	'R1':greedy_fit_obj.curve_fit_R[:,0,0],'R2':greedy_fit_obj.curve_fit_R[:,0,1],'R3':greedy_fit_obj.curve_fit_R[:,0,2],\
	# 	'R4':greedy_fit_obj.curve_fit_R[:,1,0],'R5':greedy_fit_obj.curve_fit_R[:,1,1],'R6':greedy_fit_obj.curve_fit_R[:,1,2],\
	# 	'R7':greedy_fit_obj.curve_fit_R[:,2,0],'R8':greedy_fit_obj.curve_fit_R[:,2,1],'R9':greedy_fit_obj.curve_fit_R[:,2,2]})
	# df.to_csv('curve_fit.csv',header=True,index=False)
	# df=DataFrame({'j1':greedy_fit_obj.curve_fit_js[:,0],'j2':greedy_fit_obj.curve_fit_js[:,1],'j3':greedy_fit_obj.curve_fit_js[:,2],'j4':greedy_fit_obj.curve_fit_js[:,3],'j5':greedy_fit_obj.curve_fit_js[:,4],'j6':greedy_fit_obj.curve_fit_js[:,5]})
	# df.to_csv('curve_fit_js.csv',header=False,index=False)

if __name__ == "__main__":
	main()

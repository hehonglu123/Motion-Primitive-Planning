import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *

def pose_opt(robot,curve,curve_normal,base_offset=0):

	###get curve center and best place to center in robot frame
	C=np.average(curve,axis=0)
	p=robot.fwd(np.zeros(len(robot.joint_vel_limit))).p
	#arbitrary best center point for curve
	p[0]=p[0]/2
	p[-1]=(p[-1]+base_offset)/2
	###determine Vy by eig(cov)
	curve_cov=np.cov(curve.T)
	eigenValues, eigenVectors = np.linalg.eig(curve_cov)
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	Vy=eigenVectors[0]
	###get average curve normal
	N=np.sum(curve_normal,axis=0)
	N=N/np.linalg.norm(N)
	###project N on to plane of Vy
	N=VectorPlaneProjection(N,Vy)

	###form transformation
	R=np.vstack((np.cross(Vy,-N),Vy,-N))
	T=p-R@C

	return H_from_RT(R,T)

def curve_frame_conversion(curve,curve_normal,H):
	curve_base=np.zeros(curve.shape)
	curve_normal_base=np.zeros(curve_normal.shape)

	for i in range(len(curve_base)):
		curve_base[i]=np.dot(H,np.hstack((curve[i],[1])).T)[:-1]

	#convert curve direction to base frame
	curve_normal_base=np.dot(H[:3,:3],curve_normal.T).T

	return curve_base,curve_normal_base

def find_js(robot,curve,curve_normal):
	###find all possible inv solution for given curve

	###get R first 
	curve_R=[]
	for i in range(len(curve)):
		try:
			R_curve=direction2R(curve_normal[i],-curve[i+1]+curve[i])
		except:
			# traceback.print_exc()
			pass	
		curve_R.append(R_curve)

	###insert initial orientation
	curve_R.insert(0,curve_R[0])

	###get all possible initial config
	try:
		# print(curve[0],curve_R[0])
		q_inits=np.array(robot.inv(curve[0],curve_R[0]))
	except:
		print('no init solution available')
		return

	# print(np.degrees(q_inits))
	curve_js_all=[]
	for q_init in q_inits:
		curve_js=np.zeros((len(curve),6))
		curve_js[0]=q_init
		for i in range(1,len(curve)):
			q_all=np.array(robot.inv(curve[i],curve_R[i]))
			if len(q_all)==0:
				#if no solution
				print(i)
				print(np.degrees(curve_js[i-1]))
				print('no solution available')
				return

			temp_q=q_all-curve_js[i-1]
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			if np.linalg.norm(q_all[order[0]]-curve_js[i-1])>0.5:
				print('large change')
				break	#if large changes in q
			else:
				curve_js[i]=q_all[order[0]]

		#check if all q found
		if np.linalg.norm(curve_js[-1])>0:
			curve_js_all.append(curve_js)

	return curve_js_all


def main():
	#select dataset
	data_dir='curve_1/'
	output_dir=data_dir+'baseline_motoman/'

	###read in curves
	curve = read_csv(data_dir+"Curve_dense.csv",header=None).values
	lam=calc_lam_cs(curve)

	# robot=robot_obj('ABB_6640_180_255','../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../config/paintgun.csv',d=50,acc_dict_path='')

	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun2.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=50)


	print("OPTIMIZING ON CURVE POSE")
	# H=pose_opt(robot,curve[:,:3],curve[:,3:])
	H=pose_opt(robot,curve[:,:3],curve[:,3:],base_offset=-550)
	print(H)
	
	curve_base,curve_normal_base=curve_frame_conversion(curve[:,:3],curve[:,3:],H)
	# visualize_curve_w_normal(curve_base,curve_normal_base,equal_axis=True)

	###get all inv solutions
	print("FIND ALL POSSIBLE INV SOLUTIONS")
	curve_js_all=find_js(robot,curve_base,curve_normal_base)
	print('num solutions available: ',len(curve_js_all))
	if len(curve_js_all)==0:
		return

	###get best with max(min(J_sing))
	print("FIND BEST CURVE_JS ON MIN(J_SINGULAR)")
	J_min=[]
	for i in range(len(curve_js_all)):
		J_min.append(find_j_min(robot,curve_js_all[i]))

	J_min=np.array(J_min)
	plt.figure()
	for i in range(len(J_min)):
		plt.plot(lam,J_min[i],label="inv choice "+str(i))

	plt.legend()
	plt.title('Minimum J_SINGULAR')
	plt.show()
	curve_js=curve_js_all[np.argmin(J_min.min(axis=1))]

	###save file
	np.savetxt(output_dir+'Curve_in_base_frame.csv',np.hstack((curve_base,curve_normal_base)),delimiter=',')
	np.savetxt(output_dir+'Curve_js.csv',curve_js,delimiter=',')
	np.savetxt(output_dir+'curve_pose.csv',H,delimiter=',')


if __name__ == "__main__":
	main()
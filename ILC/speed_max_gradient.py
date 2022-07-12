import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
from scipy.signal import find_peaks
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
from ilc_toolbox import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	dataset='wood/'
	data_dir="../data/"+dataset

	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	multi_valley_threshold=0.2
	robot=abb6640(d=50)

	v=250
	s = speeddata(v,9999999,9999999,999999)
	z = z10


	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('max_gradient/curve1_250_100L_multipeak/command.csv')

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	max_error=999
	inserted_points=[]
	iteration=50
	for i in range(iteration):

		ms = MotionSend()
		###execution with plant
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
		

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		print(max(error),min(speed))
		#############################error valley detection###############################
		valley=np.argmin(speed)
		valleys=[valley]


		##############################plot error#####################################
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.scatter(lam[valleys],speed[valleys],label='valleys')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax2.axis(ymin=0,ymax=2)
		ax1.axis(ymin=0,ymax=1.2*v)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		ax1.legend(loc="upper right")

		ax2.legend(loc="upper left")

		plt.legend()
		plt.savefig('recorded_data/iteration_ '+str(i))
		plt.clf()
		# plt.show()

		###########################plot for verification###################################
		# plt.figure()
		# ax = plt.axes(projection='3d')
		# ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
		# ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
		# p_bp_np=np.array([item[0] for item in p_bp])      ###np version, avoid first and last extended points
		# ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
		# ax.scatter(curve_exe[max_error_idx,0], curve_exe[max_error_idx,1], curve_exe[max_error_idx,2],c='orange',label='worst case')
		
		
		##########################################calculate gradient for speed valleys######################################
		###find closest breakpoint
		p_bp_np=np.array(p_bp).squeeze()
		_,bp_idx=calc_error(curve_exe[valley],p_bp_np)

		dv=[]
		delta=0.05
		for m in range(6):
			#tweak breakpoints
			q_bp_temp=copy.deepcopy(q_bp)
			p_bp_temp=copy.deepcopy(p_bp)
			q_bp_temp[bp_idx][0][m]+=delta
			p_bp_temp[bp_idx][0]=robot.fwd(q_bp_temp[bp_idx][0]).p
			###execution with plant
			logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp_temp,q_bp_temp,s,z)

			StringData=StringIO(logged_data)
			df = read_csv(StringData, sep =",")
			##############################data analysis#####################################
			lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.logged_data_analysis(robot,df,realrobot=True)
			#############################chop extension off##################################
			_, _, _,_, speed_temp, _=ms.chop_extension(curve_exe_temp, curve_exe_R_temp, curve_exe_js_temp, speed_temp, timestamp_temp,curve[0,:3],curve[-1,:3])

			dv.append(np.min(speed_temp)-speed[valley])

		alpha=1
		dv_dq=np.array(dv)/delta
		dv_dq=np.reshape(dv_dq,(-1,1))
		point_adjustment=-alpha*np.linalg.pinv(dv_dq)*(v-speed[valley])
		
		q_bp[bp_idx][0]+=point_adjustment[0]

if __name__ == "__main__":
	main()
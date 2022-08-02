import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
from scipy.signal import find_peaks
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../')
sys.path.append('../../toolbox')
from ilc_toolbox import *

from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	dataset='wood/'
	data_dir="../../data/"+dataset
	# fitting_output="../../data/"+dataset+'baseline/100L/'
	fitting_output="../../data/"+dataset+'qp/'
	# fitting_output="../greedy_fitting/greedy_output/curve1_movel_0.1/"
	# fitting_output="../greedy_fitting/greedy_output/curve2_movel_0.1/"


	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	robot=abb6640(d=50)

	v=250
	s = speeddata(v,9999999,9999999,999999)
	z = z10


	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('curve1_250_100L_multipeak/command.csv')

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	iteration=50
	for i in range(iteration):

		p_bp_np=np.array(p_bp)  

		ms = MotionSend()
		###execution with plant
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
		# Write log csv to file
		with open("recorded_data/curve_exe_v"+str(v)+"_z10.csv","w") as f:
			f.write(logged_data)

		ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)

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
		valleys=[np.argmin(speed)]

		##############################plot error#####################################

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.scatter(lam[valleys],error[valleys],label='valleys')
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

		##########################################calculate gradient for valleys######################################
		for valley in valleys:


			print('calculating gradient')
			dv_dq=ilc.get_speed_gradient_from_traj_exe(p_bp,q_bp,curve_exe,curve_exe_js,timestamp,lam,valley,v)
			p_bp, q_bp=ilc.update_bp_speed(p_bp,q_bp,dv_dq,speed[valley],breakpoint_interp_2tweak_indices,v)


if __name__ == "__main__":
	main()
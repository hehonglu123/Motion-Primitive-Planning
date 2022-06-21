########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO

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
	robot=abb6640(d=50)
	# data_dir="fitting_output_new/python_qp_movel/"
	dataset='from_NX/'
	data_dir="../data/"+dataset
	fitting_output="../data/"+dataset+'baseline/100L/'


	curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
	lam_original=calc_lam_cs(curve[:,:3])

	max_error_threshold=0.1
	

	s = speeddata(800,9999999,9999999,999999)
	z = z10


	curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values

	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')

	lam_bp=lam_original[breakpoints[1:]-1]

	###assume bp on original curve
	curve_bp=copy.deepcopy(p_bp)

	###extension
	p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)


	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	iteration=10
	for i in range(iteration):
		ms = MotionSend()
		###execute,curve_fit_js only used for orientation
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

		##############################calcualte max error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		
		#######################################ILC######################################
		###calculate stochastic gradient
		curve_blended_downsampled,G=ilc.sto_gradient_from_model(p_bp,q_bp,total_points=100,K=100)
		

		###find error at each point
		idx_original=[]
		idx_exe=[]
		error_downsampled=[]
		for p_blended in curve_blended_downsampled:
			idx_exe.append(calc_error(p_blended,curve_exe)[1])
			idx_original.append(calc_error(curve_exe[idx_exe[-1]],curve[:,:3])[1])
			error_downsampled.append(curve_exe[idx_exe[-1]]-curve[idx_original[-1],:3])

		print(error_downsampled)
		# plt.figure()
		# plt.plot(np.linalg.norm(error_downsampled,axis=-1))

		alpha=1/np.sqrt(i+1)
		d_bp_all=-alpha*np.reshape(G@np.array(error_downsampled).flatten(),(len(p_bp),3))
		print(d_bp_all)

		for m in range(len(d_bp_all)):
			p_bp[m][0]=p_bp[m][0]+d_bp_all[m]

		###############################plot gradient G##############################
		# plt.figure()
		# im=plt.imshow(G, cmap='hot', interpolation='nearest')
		# plt.colorbar(im)
		# plt.title("Stochastic Gradient from Analytical Model")
		# plt.xlabel('d_bp')
		# plt.ylabel('d_p_model')

		##############################plot error#####################################
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		ax1.legend(loc=0)

		ax2.legend(loc=0)


		plt.legend()

		# plt.show()
		plt.savefig('iteration '+str(i))
		plt.clf()
if __name__ == "__main__":
	main()
import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from exe_comparison import *


def main():
	ms = MotionSend()
	data_dir="fitting_output_new/all_theta_opt_blended/"
	# speed={"v50":v50,"v500":v500,"v5000":v5000}
	# zone={"fine":fine,"z1":z1,"z10":z10}
	vmax = speeddata(10000,9999999,9999999,999999)


	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv(data_dir+'arm1.csv', names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	primitives=['movej_fit']*len(curve_js)
	
	

	curve_exe_js=ms.exec_motions(primitives,[],curve_js,data_dir+'arm1.csv',vmax,z10)

	f = open(data_dir+"curve_exe"+"_vmax_z10.csv", "w")
	f.write(curve_exe_js)
	f.close()

if __name__ == "__main__":
	main()
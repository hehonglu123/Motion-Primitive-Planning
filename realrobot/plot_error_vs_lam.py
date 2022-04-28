import numpy as np
from general_robotics_toolbox import *
from pandas import *
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
import matplotlib.pyplot as plt

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *


col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

robot=abb6640(d=50)


# data_dir="fitting_output/threshold0.1/"
# speed={"v50":v50,"v500":v500,"v5000":v5000}
# zone={"fine":fine,"z1":z1,"z10":z10}
speed=['v50','v150','v200','v400','v800']
zone=['z10']
data_dir="recorded_data/qp_movel/"


for s in speed:
    for z in zone:

		###read in curve_exe
		col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
		data = read_csv(data_dir+"curve_exe_"+s+'_'+z+".csv",names=col_names)
		q1=data['J1'].tolist()[1:]
		q2=data['J2'].tolist()[1:]
		q3=data['J3'].tolist()[1:]
		q4=data['J4'].tolist()[1:]
		q5=data['J5'].tolist()[1:]
		q6=data['J6'].tolist()[1:]

		cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
		start_idx=np.where(cmd_num==3)[0][0]
		timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)[start_idx:]
		curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])


		curve_exe=[]
		curve_exe_R=[]
		for i in range(len(curve_exe_js)):
			robot_pose=robot.fwd(curve_exe_js[i])
			curve_exe.append(robot_pose.p)
			curve_exe_R.append(robot_pose.R)

		lam=calc_lam_cs(curve_exe)
		# error_all=calc_all_error(curve_exe,curve)
		# error_ex_blending,lam_ex_blending=calc_all_error_ex_blending(curve_exe,curve,10,lam,np.linspace(0,lam[-1],50))

		plt.plot(lam,error_all)
		plt.scatter(lam_ex_blending,error_ex_blending)
		plt.ylim(0,3)
		plt.title(s+' Error vs Lambda')
		plt.xlabel('Lambda (mm)')
		plt.ylabel('Projected Error (mm)')
		plt.ylim(0,3)
		plt.show()
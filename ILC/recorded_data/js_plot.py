import numpy as np
from general_robotics_toolbox import *
from pandas import *
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from utils import *


robot=abb6640(d=50)



###read in recorded joint train_data
data = read_csv("noise_from_RS.csv")
q1=data[' J1'].tolist()
q2=data[' J2'].tolist()
q3=data[' J3'].tolist()
q4=data[' J4'].tolist()
q5=data[' J5'].tolist()
q6=data[' J6'].tolist()
cmd_num=np.array(data[' cmd_num'].tolist()).astype(float)
start_idx=np.where(cmd_num==4)[0][0]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)
timestep=np.average(timestamp[1:]-timestamp[:-1])

lam2=calc_lam_js(curve_exe_js,robot)


for i in range(6):
	plt.figure(i)
	plt.plot(lam2,curve_exe_js[:,i],label='execution')
	plt.legend()

	plt.title('Joints Plot')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv, read_excel
import sys
sys.path.append('../../../toolbox')
from abb_motion_program_exec_client import *
from robots_def import *
from lambda_calc import *

data_dir='fitting_output_new/all_theta_opt/'

col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data=read_csv(data_dir+'curve_fit_js.csv',names=col_names)
q1=data['J1'].tolist()
q2=data['J2'].tolist()
q3=data['J3'].tolist()
q4=data['J4'].tolist()
q5=data['J5'].tolist()
q6=data['J6'].tolist()
curve_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)


# col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
# data=read_csv(data_dir+'curve_exe_vmax_z10.csv',names=col_names)
# q1=data['J1'].tolist()[1:]
# q2=data['J2'].tolist()[1:]
# q3=data['J3'].tolist()[1:]
# q4=data['J4'].tolist()[1:]
# q5=data['J5'].tolist()[1:]
# q6=data['J6'].tolist()[1:]
# timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)
# cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
# start_idx=np.where(cmd_num==5)[0][0]
# curve_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])

robot=abb6640(d=50)
lam=[0]
curve_exe=[]
curve_exe_R=[]
for i in range(len(curve_js)):
    robot_pose=robot.fwd(curve_js[i])
    curve_exe.append(robot_pose.p)
    curve_exe_R.append(robot_pose.R)
    if i>0:
        lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))

lamdot=calc_lamdot(curve_js,lam,robot,1)
plt.plot(lam[1:],lamdot)
plt.title("lambdadot vs lambda")
plt.ylabel('lambdadot (mm/s)')
plt.xlabel('lambda (mm)')
plt.show()
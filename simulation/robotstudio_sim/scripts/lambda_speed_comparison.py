import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv, read_excel
import sys
sys.path.append('../../../toolbox')
from abb_motion_program_exec_client import *
from robots_def import *
from lambda_calc import *

# data_dir='fitting_output_new/Jon/'
data_dir="fitting_output_new/curve_pose_opt_blended/"
robot=abb6640(d=50)

# col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
# data=read_csv(data_dir+'qbestcurve.csv',names=col_names)
# q1=data['J1'].tolist()
# q2=data['J2'].tolist()
# q3=data['J3'].tolist()
# q4=data['J4'].tolist()
# q5=data['J5'].tolist()
# q6=data['J6'].tolist()
# curve_js_opt=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)

# lam_opt=[0]
# curve_opt=[]
# for i in range(len(curve_js_opt)):
#     robot_pose=robot.fwd(curve_js_opt[i])
#     curve_opt.append(robot_pose.p)
#     if i>0:
#         lam_opt.append(lam_opt[-1]+np.linalg.norm(curve_opt[i]-curve_opt[i-1]))

# lamdot_opt=calc_lamdot(curve_js_opt,lam_opt,robot,1)


col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data=read_csv(data_dir+'curve_fit_js.csv',names=col_names)
q1=data['J1'].tolist()
q2=data['J2'].tolist()
q3=data['J3'].tolist()
q4=data['J4'].tolist()
q5=data['J5'].tolist()
q6=data['J6'].tolist()
curve_js_fit=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)

lam_fit=[0]
curve_fit=[]
for i in range(len(curve_js_fit)):
    robot_pose=robot.fwd(curve_js_fit[i])
    curve_fit.append(robot_pose.p)
    if i>0:
        lam_fit.append(lam_fit[-1]+np.linalg.norm(curve_fit[i]-curve_fit[i-1]))

lamdot_fit=calc_lamdot(curve_js_fit,lam_fit,robot,1)




speed='vmax'
zone='z10'
###read in curve_exe
col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
df = read_csv(data_dir+"curve_exe"+"_"+s+"_"+z+".csv")
lam, curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp=ms.logged_data_analysis(robot,df)


# plt.plot(lam_opt[1:],lamdot_opt, label='Optimization')

plt.plot(lam_fit,lamdot_fit, label='Fitting')

plt.plot(lam,lamdot_act, label='Recorded Joints')
plt.plot(lam[1:],act_speed, label='Actual speed')

plt.title("speed vs lambda")
plt.ylabel('speed (mm/s)')
plt.xlabel('lambda (mm)')
plt.legend()
plt.show()
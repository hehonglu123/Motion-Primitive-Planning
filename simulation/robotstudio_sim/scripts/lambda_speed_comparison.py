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
robot=abb6640(d=50)

col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data=read_csv(data_dir+'arm1.csv',names=col_names)
q1=data['J1'].tolist()
q2=data['J2'].tolist()
q3=data['J3'].tolist()
q4=data['J4'].tolist()
q5=data['J5'].tolist()
q6=data['J6'].tolist()
curve_js_opt=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)

lam_opt=[0]
curve_opt=[]
for i in range(len(curve_js_opt)):
    robot_pose=robot.fwd(curve_js_opt[i])
    curve_opt.append(robot_pose.p)
    if i>0:
        lam_opt.append(lam_opt[-1]+np.linalg.norm(curve_opt[i]-curve_opt[i-1]))

lamdot_opt=calc_lamdot(curve_js_opt,lam_opt,robot,1)


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


col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data=read_csv(data_dir+'curve_exe_vmax_z10.csv',names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)
cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
start_idx=np.where(cmd_num==5)[0][0]
curve_js_act=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])

lam_act=[0]
curve_act=[]
for i in range(len(curve_js_act)):
    robot_pose=robot.fwd(curve_js_act[i])
    curve_act.append(robot_pose.p)
    if i>0:
        lam_act.append(lam_act[-1]+np.linalg.norm(curve_act[i]-curve_act[i-1]))

lamdot_act=calc_lamdot(curve_js_act,lam_act,robot,1)



speed='vmax'
zone='z10'
###read in curve_exe
col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data = read_csv(data_dir+"curve_exe_"+speed+'_'+zone+".csv",names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
start_idx=np.where(cmd_num==5)[0][0]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)

timestep=np.average(timestamp[1:]-timestamp[:-1])

act_speed=[]
lam_speed=[0]
curve_exe=[]
curve_exe_R=[]
for i in range(len(curve_exe_js)):
    robot_pose=robot.fwd(curve_exe_js[i])
    curve_exe.append(robot_pose.p)
    curve_exe_R.append(robot_pose.R)
    if i>0:
        lam_speed.append(lam_speed[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
    try:
        if timestamp[-1]!=timestamp[-2]:
            act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
            
    except IndexError:
        pass


plt.plot(lam_opt[1:],lamdot_opt, label='Optimization')

plt.plot(lam_fit[1:],lamdot_fit, label='Fitting')

plt.plot(lam_act[1:],lamdot_act, label='Actual Joints')
plt.plot(lam_speed[1:],act_speed, label='Actual speed')

plt.title("speed vs lambda")
plt.ylabel('speed (mm/s)')
plt.xlabel('lambda (mm)')
plt.legend()
plt.show()
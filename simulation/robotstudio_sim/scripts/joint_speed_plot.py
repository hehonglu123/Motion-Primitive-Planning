import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *

speed='vmax'
zone='z10'
data_dir='fitting_output_new/init_opt/'
###read in curve_exe
col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data = read_csv(data_dir+"curve_exe_"+speed+'_'+zone+".csv",names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)
cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
start_idx=np.where(cmd_num==3)[0][0]
curve_exe_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:]
qdot=[]
qddot=[]
curve_exe=[]
lam=[0]
robot=abb6640(d=50)
for i in range(len(curve_exe_js)-1):
    robot_pose=robot.fwd(np.radians(curve_exe_js[i]))
    curve_exe.append(robot_pose.p)
    if i>0:
        lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))

    qdot.append((curve_exe_js[i+1]-curve_exe_js[i])/(timestamp[i+1]-timestamp[i]))
    if i!=0:
        qddot.append((qdot[-1]-qdot[-2])/(timestamp[i+1]-timestamp[i]))

qdot=np.array(qdot)
qddot=np.array(qddot)
plt.figure(1)
plt.plot(lam,qdot[:,0], label="q1dot")
plt.plot(lam,qdot[:,1], label="q2dot")
plt.plot(lam,qdot[:,2], label="q3dot")
plt.plot(lam,qdot[:,3], label="q4dot")
plt.plot(lam,qdot[:,4], label="q5dot")
plt.plot(lam,qdot[:,5], label="q6dot")


plt.title("Joint Speed: "+data_dir+speed+'_'+zone)
plt.legend()
plt.ylim([-200,200])
plt.ylabel('Speed (mm/s)')
plt.xlabel('lambda (mm)')

plt.figure(2)
plt.plot(lam[:-1],qddot[:,0], label="q1ddot")
plt.plot(lam[:-1],qddot[:,1], label="q2ddot")
plt.plot(lam[:-1],qddot[:,2], label="q3ddot")
plt.plot(lam[:-1],qddot[:,3], label="q4ddot")
plt.plot(lam[:-1],qddot[:,4], label="q5ddot")
plt.plot(lam[:-1],qddot[:,5], label="q6ddot")
plt.title("Joint Acceleration: "+data_dir+speed+'_'+zone)
plt.legend()
# plt.ylim([-3000,3000])
plt.ylabel('Speed (mm/s)')
plt.xlabel('lambda (mm)')
plt.show()
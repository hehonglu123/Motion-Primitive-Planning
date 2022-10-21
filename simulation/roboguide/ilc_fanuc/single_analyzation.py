from copy import deepcopy
import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from scipy.signal import find_peaks
import yaml
from matplotlib import pyplot as plt
from pathlib import Path
from fanuc_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *


curve_folder = '../data/baseline_m10ia/blade_scale/'
curve_js_folder = curve_folder+'/25/results_243/'
robot=m10ia(d=50)

##### read curve
curve = read_csv(curve_folder+"Curve_in_base_frame.csv",header=None).values

##### curve js from robot
ms = MotionSendFANUC(robot_ip='127.0.0.3')
try:
    breakpoints,primitives,p_bp,q_bp,_=ms.extract_data_from_cmd(os.getcwd()+'/'+curve_js_folder+'command_arm1_3.csv')
    # breakpoints,primitives,p_bp,q_bp=extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_25.csv')
except:
    print("Read command error")
    exit()
# primitives,p_bp,q_bp=ms.extend_start_end(robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100)
# s=152
# z=100
# ###execute,curve_fit_js only used for orientation
# logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
# StringData=StringIO(logged_data.decode('utf-8'))
# df = read_csv(StringData)
# print(df)
# ##############################data analysis#####################################
# lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
# #############################chop extension off##################################
# lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve[:,3:])
# df=DataFrame({'timestamp':timestamp,'q0':curve_exe_js[:,0],'q1':curve_exe_js[:,1],'q2':curve_exe_js[:,2],'q3':curve_exe_js[:,3],'q4':curve_exe_js[:,4],'q5':curve_exe_js[:,5]})
# df.to_csv(curve_js_folder+'curve_js_exe.csv',header=False,index=False)

# ######## curve js from file
with open(curve_js_folder+'curve_js_exe.csv',"rb") as f:
    curve_exe_js=read_csv(f,header=None).values
timestamp=curve_exe_js[:,0]
curve_exe_js=curve_exe_js[:,1:]

curve_exe=[]
curve_exe_R=[]
lam=[0]
speed=[]
for i in range(len(curve_exe_js)):
    this_p = robot.fwd(curve_exe_js[i])
    curve_exe.append(this_p.p)
    curve_exe_R.append(this_p.R)
    if i>0:
        lam.append(lam[-1]+np.linalg.norm(curve_exe[-1]-curve_exe[-2]))
        speed.append(np.fabs(lam[-1]-lam[-2])/(timestamp[i]-timestamp[i-1]))
speed.append(speed[-1])
curve_exe=np.array(curve_exe)
curve_exe_R=np.array(curve_exe_R)
lam=np.array(lam)

error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

ana_range=[0]
for i in range(2,len(error)-2):
    if error[i]<error[i-2] and error[i]<error[i-1] and \
        error[i]<error[i+1] and error[i]<error[i+2]:
        ana_range.append(i)
ana_range.append(len(error)-1)

fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.plot(lam, speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
# ax1.plot(np.arange(len(lam)), speed, 'g-', label='Speed')
# ax2.plot(np.arange(len(lam)), error, 'b-',label='Error')
# ax2.plot(np.arange(len(lam)), np.degrees(angle_error), 'y-',label='Normal Error')
ax2.scatter(lam[ana_range],error[ana_range],label='Error valley')
draw_speed_max=max(speed)*1.05
draw_error_max=max(error)*1.05
ax1.axis(ymin=0,ymax=draw_speed_max)
ax2.axis(ymin=0,ymax=draw_error_max)
ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot")
ax1.legend(loc=0)
ax2.legend(loc=0)
plt.show()

p_bp_sq=np.squeeze(p_bp)
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='execution')
ax.scatter3D(curve_exe[ana_range,0], curve_exe[ana_range,1],curve_exe[ana_range,2], 'blue', label='Error valley')   
ax.scatter3D(p_bp_sq[:,0], p_bp_sq[:,1],p_bp_sq[:,2], 'red', label='break points')
plt.legend()
plt.show()           

import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
from MotionSend import *

robot=abb6640(d=50)
#determine correct commanded speed to keep error within 1mm

num_ls=[100]

dataset="from_NX/"
curve_js = read_csv(dataset+'Curve_js.csv',header=None).values
curve = read_csv(dataset+"Curve_in_base_frame.csv",header=None).values
step=10
lam=calc_lam_cs(curve)
lamdot=calc_lamdot(curve_js,lam,robot,step)

ms=MotionSend()
for num_l in num_ls:
    ms = MotionSend()
    data_dir=dataset+"baseline/"+str(num_l)+'L/'
    df = read_csv(data_dir+"curve_exe_v402.83203125_z10.csv")
    lam_exe, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)

    error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)

    start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
    end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

    curve_exe=curve_exe[start_idx:end_idx+1]
    curve_exe_R=curve_exe_R[start_idx:end_idx+1]
    curve_exe_js=curve_exe_js[start_idx:end_idx+1]
    speed=speed[start_idx:end_idx+1]
    lam_exe=calc_lam_cs(curve_exe)
    
    lamdot_exe=calc_lamdot(curve_exe_js,lam_exe,robot,step)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(lam_exe, speed, 'g-', label='Speed')
    ax2.plot(lam_exe, error, 'b-',label='Error')
    ax2.plot(lam_exe, np.degrees(angle_error), 'y-',label='Normal Error')

    ax1.set_xlabel('Lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Execution Results")
    ax1.legend(loc=0)

    ax2.legend(loc=0)

    plt.legend()

    plt.figure()
    plt.plot(lam_exe[::step],lamdot_exe,label='execution curve')
    plt.plot(lam[::step],lamdot,label='original curve')
    plt.xlabel('Lambda (mm)')
    plt.ylabel('Lambdadot Constraint (mm/s)')
    plt.legend()
    plt.title('Lambda Dot Constraint')
    plt.show()

    


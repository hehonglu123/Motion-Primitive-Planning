import numpy as np
from scipy.signal import find_peaks
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from matplotlib import pyplot as plt
from pathlib import Path
import os
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

def main():
    # data_dir="fitting_output_new/python_qp_movel/"
    dataset='blade/'
    # dataset='wood/'
    data_dir="data/"+dataset
    fitting_output='data/'+dataset
    

    curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    curve_normal=curve[:,3:]
    
    multi_peak_threshold=0.2
    robot=m710ic(d=50)

    s = 1300
    z = 100
    alpha = 0.5 # for gradient descent
    ilc_output=fitting_output+'results_'+str(s)+'/'
    Path(ilc_output).mkdir(exist_ok=True)

    ###TODO: extension fix start point, moveC support
    ms = MotionSendFANUC()
    iteration=200
    draw_y_max=None
    max_error_tolerance = 0.5
    for i in range(iteration):
        breakpoints,primitives,p_bp,q_bp=extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_'+str(i)+'.csv')
        
        ###execute,curve_fit_js only used for orientation
        logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

        # with open(data_dir+"curve_js_iter0_exe.csv","wb") as f:
        #     f.write(logged_data)
        # exit()

        # print(logged_data)
        StringData=StringIO(logged_data.decode('utf-8'))
        # df = read_csv(StringData, sep =",")
        df = read_csv(StringData)
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)

        #############################chop extension off##################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve[:,3:])
        ave_speed=np.mean(speed)

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        print('Iteration:',i,', Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
        print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
        print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
        print("===========================================")
        #############################error peak detection###############################
        find_peak_dist = 20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1])
        if find_peak_dist<1:
            find_peak_dist=1
        peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks

        if len(peaks)==0 or np.argmax(error) not in peaks:
            peaks=np.append(peaks,np.argmax(error))
        
        ##############################plot error#####################################
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax1.axis(ymin=0,ymax=800)
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.scatter(lam[peaks],error[peaks],label='peaks')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
        if draw_y_max is None:
            draw_y_max=max(error)*1.05
        # draw_y_max=2.5
        ax2.axis(ymin=0,ymax=draw_y_max)

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        ax1.legend(loc=1)
        ax2.legend(loc=2)
        plt.title("Speed and Error Plot")

        # save fig
        # plt.legend()
        plt.savefig(ilc_output+'iteration_'+str(i))
        plt.clf()
        # plt.show()
        # exit()
        if max(error) < max_error_tolerance:
            break

if __name__ == "__main__":
    main()
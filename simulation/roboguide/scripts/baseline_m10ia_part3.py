from math import ceil, radians, floor, degrees
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from io import StringIO
from general_robotics_toolbox import *
import sys

# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
from fanuc_motion_program_exec_client import *

# define m900ia
robot=m10ia(d=50)
client = FANUCClient()
ms = MotionSendFANUC()
utool_num = 2

# all_objtype=['wood','blade_scale']
all_objtype=['blade_scale']
# all_objtype=['wood']

# num_ls=[80,100,150]
num_ls=[100,50,30,25]

for obj_type in all_objtype:

    # obj_type='wood'
    # obj_type='blade'
    print(obj_type)
    
    data_dir='../data/baseline_m10ia/'+obj_type+'/'

    robot=m10ia(d=50)
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    curve = np.array(curve)
    curve_normal = curve[:,3:]
    curve = curve[:,:3]
    ###read actual curve
    # curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values
    # curve_js=np.array(curve_js)

    for num_l in num_ls:
        print(obj_type+' '+str(num_l))
        breakpoints,primitives,p_bp,q_bp,_=ms.extract_data_from_cmd(data_dir+str(num_l)+'/command.csv')

        q_bp_start = q_bp[0][0]
        q_bp_end = q_bp[-1][-1]
        primitives,p_bp,q_bp=ms.extend_start_end(robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100)

        ## calculate step at start and end
        step_start1=None
        step_end1=None
        for i in range(len(q_bp)):
            if np.all(q_bp[i][0]==q_bp_start):
                step_start1=i
            if np.all(q_bp[i][-1]==q_bp_end):
                step_end1=i

        assert step_start1 is not None,'Cant find step start'
        assert step_end1 is not None,'Cant find step start'
        print(step_start1,step_end1)

        s_up=1000
        s_low=0
        s=(s_up+s_low)/2
        speed_found=False
        z=100
        last_error = 999
        find_sol=False
        while True:
            s=int(s)
            
            logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
            StringData=StringIO(logged_data.decode('utf-8'))
            df = read_csv(StringData)
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve[:,3:])
            speed_ave=np.mean(speed)
            speed_std=np.std(speed)
            error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve_normal)
            error_max=max(error)
            angle_error_max=max(angle_error)

            if speed_found:
                break

            print('Speed:',s,',Error:',error_max,',Angle Error:',degrees(angle_error_max),'Speed Ave:',speed_ave,'Speed Std:',speed_std)
            if error_max > 0.5 or angle_error_max > radians(3) or speed_std>speed_ave*0.05:
                s_up=s
                s=ceil((s_up+s_low)/2.)
            else:
                s_low=s
                s=floor((s_up+s_low)/2.)
                find_sol=True
            if s==s_low:
                break
            elif s==s_up:
                s=s_low
                speed_found=True

            if not find_sol:
                print("last error:",last_error,"this error:",error_max)
                if last_error-error_max < 0.001:
                    print("Can't reduce error")
                    break
            last_error=error_max
        
        lamdot_act=calc_lamdot(curve_exe_js,lam,robot,1)
        
        with open(data_dir+str(num_l)+"/curve_js_exe_"+str(s)+".csv","wb") as f:
            f.write(curve_exe_js)
        with open(data_dir+str(num_l)+'/error.npy','wb') as f:
            np.save(f,error)
        with open(data_dir+str(num_l)+'/normal_error.npy','wb') as f:
            np.save(f,angle_error)
        with open(data_dir+str(num_l)+'/speed.npy','wb') as f:
            np.save(f,speed)
        with open(data_dir+str(num_l)+'/lambda.npy','wb') as f:
            np.save(f,lam)
        with open(data_dir+str(num_l)+'/timestamps.npy','wb') as f:
            np.save(f,timestamp)
        
        print('Num_l:',num_l,', Max Error:',max(error),'Ave. Speed:',np.mean(speed),'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/np.mean(speed)*100)
        print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
        print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
        print("===========================================")

        fig, ax1 = plt.subplots(figsize=(6,4))
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
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
        # save fig
        plt.legend()
        plt.savefig(data_dir+str(num_l)+'/iteration_'+str(i))
        plt.clf()
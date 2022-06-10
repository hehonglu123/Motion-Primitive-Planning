from cProfile import label
from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos,ceil,floor
from copy import deepcopy
from pathlib import Path

from general_robotics_toolbox import *
import sys
import os
from matplotlib import pyplot as plt

from heuristic_curve_fit_func import *

sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

img_dir = 'data_imgs/'

# all_ang=[2.5,5,10,20,30]
all_ang=[30]

all_speed_ave = {}
all_speed_max = {}
all_speed_min = {}
all_speed_std = {}
all_speed_max_min_per = {}

for ang in all_ang:

    print("========================")
    print("Mismatch Ang:",ang)
    ang_data_dir = 'data_ang_'+str(ang)+'/'

    with open(ang_data_dir+'Curve_js.npy','rb') as f:
        curve_js = np.load(f)
    with open(ang_data_dir+'Curve_in_base_frame.npy','rb') as f:
        curve = np.load(f)
    with open(ang_data_dir+'Curve_R_in_base_frame.npy','rb') as f:
        R_all = np.load(f)
        curve_normal=R_all[:,:,-1]

    # all_cases=['do_nothing','exploit_tolerance','add_movel','add_movec','addmid_h']
    all_cases=['do_nothing']
    # all_list = os.listdir(os.getcwd()+'/'+ang_data_dir)
    # all_cases=[]
    # for obj in all_list:
    #     if os.path.isdir(ang_data_dir+obj):
    #         all_cases.append(obj)
    
    if len(all_speed_ave) == 0:
        for this_case in all_cases:
            all_speed_ave[this_case]=[]
            all_speed_max[this_case]=[]
            all_speed_min[this_case]=[]
            all_speed_std[this_case]=[]
            all_speed_max_min_per[this_case]=[]

    for this_case in all_cases:
        print('---------')
        print(this_case)
        
        data_dir=ang_data_dir+this_case+'/'

        # the executed result
        with open(data_dir+'error.npy','rb') as f:
            error = np.load(f)
        with open(data_dir+'normal_error.npy','rb') as f:
            angle_error = np.load(f)
        with open(data_dir+'speed.npy','rb') as f:
            act_speed = np.load(f)
        with open(data_dir+'lambda.npy','rb') as f:
            lam_exec = np.load(f)
        
        act_speed = np.append(0,act_speed)
        poly = np.poly1d(np.polyfit(lam_exec,act_speed,deg=40))
        poly_der = np.polyder(poly)
        # fit=poly(lam_exec[1:])     
        start_id=10
        end_id=-10
        while True:
            if poly_der(lam_exec[start_id]) < 0.5:
                break
            start_id+=1
        while True:
            if poly_der(lam_exec[end_id]) > -0.5:
                break
            end_id -= 1

        # start_id=0
        # end_id=-1
        # while True:
        #     last_start_id=start_id
        #     last_end_id=end_id
        #     while True:
        #         if act_speed[start_id] >= np.mean(act_speed[last_start_id:last_end_id]):
        #             break
        #         start_id+=1
        #     while True:
        #         if act_speed[end_id-1] >= np.mean(act_speed[last_start_id:last_end_id]):
        #             break
        #         end_id-=1
        #     if last_start_id==start_id and last_end_id==end_id:
        #         break
        
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(lam_exec,act_speed, 'g-', label='Speed')
        # ax1.plot(lam_exec[start_id:end_id],act_speed[start_id:end_id], 'r-', label='Speed')
        ax2.plot(lam_exec, error, 'b-',label='Error')
        # ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Execution Result ")
        ax1.legend(loc=6)
        ax2.legend(loc=7)
        # plt.show()
        plt.savefig(img_dir+'speed_error_ang_'+str(ang)+'_'+this_case+'.png')
        plt.clf()

        error=error[start_id:end_id]
        angle_error=angle_error[start_id:end_id]
        act_speed=act_speed[start_id:end_id]
        lam_exec=lam_exec[start_id:end_id]

        

        print("Max Speed:",np.max(act_speed),"Min Speed:",np.min(act_speed),"Ave Speed:",np.mean(act_speed),"Speed Std:",np.std(act_speed),"Max Min %:",(np.max(act_speed)-np.min(act_speed))*100/np.max(act_speed))
        print("Ave Err:",np.mean(error),"Max Err:",np.max(error),"Min Err:",np.min(error),"Err Std:",np.std(error))
        print("Norm Ave Err:",np.mean(angle_error),"Norm Max Err:",np.max(angle_error),"Norm Min Err:",np.min(angle_error),"Norm Err Std:",np.std(angle_error))
        print("Norm Ave Err:",degrees(np.mean(angle_error)),"Norm Max Err:",degrees(np.max(angle_error)),"Norm Min Err:",degrees(np.min(angle_error)),"Norm Err Std:",degrees(np.std(angle_error)))
        print("======================")

        all_speed_ave[this_case].append(np.mean(act_speed))
        all_speed_max[this_case].append(np.max(act_speed))
        all_speed_min[this_case].append(np.min(act_speed))
        all_speed_std[this_case].append(np.std(act_speed)*100/np.mean(act_speed))
        all_speed_max_min_per[this_case].append((np.max(act_speed)-np.min(act_speed))*100/np.max(act_speed))

def plot_data(data,title,xlabel,ylabel,save_name):
    for this_case in data.keys():
        plt.plot(all_ang,data[this_case],'o-',label=this_case)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_name)
    plt.clf()


plot_data(all_speed_ave,'Ave Speed v.s. Mismatch Angle','Mismatch Angle (degree)','Ave Speed (mm/s)',img_dir+'speed_ave_ang')
plot_data(all_speed_max,'Max Speed v.s. Mismatch Angle','Mismatch Angle (degree)','Max Speed (mm/s)',img_dir+'speed_max_ang')
plot_data(all_speed_min,'Min Speed v.s. Mismatch Angle','Mismatch Angle (degree)','Min Speed (mm/s)',img_dir+'speed_min_ang')
plot_data(all_speed_std,'Speed Std over Ave Speed v.s. Mismatch Angle','Mismatch Angle (degree)','Speed Std over Ave Speed (%)',img_dir+'speed_std_ang')
plot_data(all_speed_max_min_per,'Max Min Speed Difference % v.s. Mismatch Angle','Mismatch Angle (degree)','%',img_dir+'speed_per_ang')
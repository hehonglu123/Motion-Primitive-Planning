from copy import deepcopy
from math import ceil, radians, floor, degrees
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from general_robotics_toolbox import *
import sys
import os

all_list = os.listdir(os.getcwd())
cases=[]
for obj in all_list:
    if os.path.isdir(obj):
        cases.append(obj)
print(cases)

for this_case in cases:
    print(this_case)
    data_dir = this_case+'/'
    with open(data_dir+'error.npy','rb') as f:
        error = np.load(f)
    with open(data_dir+'normal_error.npy','rb') as f:
        angle_error = np.load(f)
    with open(data_dir+'speed.npy','rb') as f:
        act_speed = np.load(f)
    with open(data_dir+'lambda.npy','rb') as f:
        lam_exec = np.load(f)

    start_id=0
    end_id=-1
    while True:
        last_start_id=start_id
        last_end_id=end_id
        while True:
            if act_speed[start_id] >= np.mean(act_speed[last_start_id:last_end_id]):
                break
            start_id+=1
        while True:
            if act_speed[end_id-1] >= np.mean(act_speed[last_start_id:last_end_id]):
                break
            end_id-=1
        if last_start_id==start_id and last_end_id==end_id:
            break
    error=error[start_id:end_id]
    angle_error=angle_error[start_id:end_id]
    act_speed=act_speed[start_id:end_id]
    lam_exec=lam_exec[start_id:end_id]

    print("Max Speed:",np.max(act_speed),"Min Speed:",np.min(act_speed),"Ave Speed:",np.mean(act_speed),"Speed Std:",np.std(act_speed))
    print("Ave Err:",np.mean(error),"Max Err:",np.max(error),"Min Err:",np.min(error),"Err Std:",np.std(error))
    print("Norm Ave Err:",np.mean(angle_error),"Norm Max Err:",np.max(angle_error),"Norm Min Err:",np.min(angle_error),"Norm Err Std:",np.std(angle_error))
    print("Norm Ave Err:",degrees(np.mean(angle_error)),"Norm Max Err:",degrees(np.max(angle_error)),"Norm Min Err:",degrees(np.min(angle_error)),"Norm Err Std:",degrees(np.std(angle_error)))

    print("======================")

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(lam_exec[1:],act_speed, 'g-', label='Speed')
    ax2.plot(lam_exec, error, 'b-',label='Error')
    # ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')

    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Execution Result "+this_case)
    # plt.title("Execution Result (Speed/Error/Normal Error v.s. Lambda)")
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.show()
    # plt.savefig(data_dir+'speed_error.png')
    # plt.clf()

    
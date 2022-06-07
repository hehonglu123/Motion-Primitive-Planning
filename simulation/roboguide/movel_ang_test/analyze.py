from cProfile import label
from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos,ceil,floor
from copy import deepcopy
from pathlib import Path
from PIL import Image
from matplotlib import image

from general_robotics_toolbox import *
import sys
from matplotlib import pyplot as plt
sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

# generate curve with linear orientation
# robot=abb6640(d=50)
robot=m900ia(d=50)
zone=100
# tolerance=1
tolerance=0.5

img_dir = 'data_img/'

all_ang=[1,2.5,5,10,20,30]
# all_speed=[1000,750,500,250,125,63,32,16,8]
# all_speed=[750,500,250,125,63,32,16]
all_speed=np.append(np.arange(10,100,10),np.arange(100,1001,100))


max_error_speed_ang=[]
max_error_ang=[]
for ang in all_ang:
    ang_data_dir='data_ang_'+str(ang)+'/'
    with open(ang_data_dir+'Curve_js.npy','rb') as f:
        curve_js = np.load(f)
    with open(ang_data_dir+'Curve_in_base_frame.npy','rb') as f:
        curve = np.load(f)
    with open(ang_data_dir+'Curve_R_in_base_frame.npy','rb') as f:
        R_all = np.load(f)
        curve_normal=R_all[:,:,-1]

    max_error_speed=[]
    max_error=[]
    speed_std = []
    speed_ave = []

    for speed in all_speed:
        data_dir=ang_data_dir+'speed_'+str(speed)+'/'

        # the executed result
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

        max_error_speed.append(act_speed[np.argmax(error)-1])
        max_error.append(np.max(error))
        speed_std.append(np.std(act_speed))
        speed_ave.append(np.mean(act_speed))

        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(lam_exec[1:],act_speed, 'g-', label='Speed')
        # # ax2.plot(lam_exec, error, 'b-',label='Error')
        # # ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')
        # ax1.set_xlabel('lambda (mm)')
        # ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        # ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        # plt.title("Execution Result ")
        # # plt.title("Execution Result (Speed/Error/Normal Error v.s. Lambda)")
        # ax1.legend(loc=0)
        # ax2.legend(loc=0)
        # plt.show()
    
    max_error_speed=np.array(max_error_speed)
    max_error=np.array(max_error)
    speed_std=np.array(speed_std)
    speed_ave=np.array(speed_ave)

    poly = np.poly1d(np.polyfit(max_error_speed,max_error,deg=1))
    fit=poly([max_error_speed[0],max_error_speed[-1]])
    # plt.scatter(max_error,max_error_speed)
    plt.scatter(max_error_speed,max_error)
    # plt.plot([max_error[0],max_error[-1]],fit,'r',label='regression line')
    plt.plot([max_error_speed[0],max_error_speed[-1]],fit,'r',label='regression line')
    plt.ylabel("Max Error (mm)")
    plt.xlabel("Speed at Max Error (mm/s)")
    plt.legend()
    plt.title('Speed v.s. Max Error, Mismatch Ang:'+str(ang))
    # plt.show()
    plt.savefig(img_dir+"error_speed_ang_"+str(ang)+'.png')
    plt.clf()

    poly = np.poly1d(np.polyfit(all_speed,np.divide(speed_ave-max_error_speed,speed_std),deg=1))
    fit=poly([all_speed[0],all_speed[-1]])
    # plt.scatter(max_error,max_error_speed)
    plt.scatter(all_speed,np.divide(speed_ave-max_error_speed,speed_std))
    # plt.plot([max_error[0],max_error[-1]],fit,'r',label='regression line')
    plt.plot([all_speed[0],all_speed[-1]],fit,'r',label='regression line')
    plt.ylabel("Speed at Max Error over Command Speed (%)")
    plt.xlabel("Command Speed (mm/s)")
    plt.legend()
    plt.title('Cmd Speed v.s. Speed at Max Error over Cmd Speed (%), Mismatch Ang:'+str(ang))
    # plt.show()
    plt.savefig(img_dir+"errspeed_cmdspeed_"+str(ang)+'.png')
    plt.clf()

    max_error_speed_ang.append(max_error_speed)
    max_error_ang.append(max_error)
    print("===========================")

# poly = np.poly1d(np.polyfit(all_ang,speed_over_error_ave,deg=1))
# rat_fit=poly([all_ang[0],all_ang[-1]])
# plt.scatter(all_ang,speed_over_error_ave)
# plt.plot([all_ang[0],all_ang[-1]],rat_fit)
# plt.show()

# exit()

max_error_speed_ang=np.array(max_error_speed_ang)
max_error_ang=np.array(max_error_ang)

i=0
corr_coef = []
for speed in all_speed:
    this_speed = max_error_speed_ang[:,i]
    this_error = max_error_ang[:,i]

    # # speed at max error v.s. mismatch angle    
    # poly = np.poly1d(np.polyfit(all_ang,this_speed,deg=1))
    # fit=poly([all_ang[0],all_ang[-1]])
    # plt.scatter(all_ang,this_speed)
    # plt.plot([all_ang[0],all_ang[-1]],fit)
    # plt.show()

    # # speed at max error over cmd speed v.s. mismatch angle
    poly = np.poly1d(np.polyfit(all_ang,this_speed,deg=1))
    fit=poly([all_ang[0],all_ang[-1]])
    plt.scatter(all_ang,this_speed)
    plt.plot([all_ang[0],all_ang[-1]],fit,'r',label='regression line')
    plt.xlabel("Mismatch Ang (deg)")
    plt.ylabel("Speed at Max Error (mm/s)")
    plt.title('Mismatch Ang v.s. Speed at Max Error, Cmd Speed:'+str(speed))
    plt.savefig(img_dir+"errspeed_ang_speed_"+str(speed)+'.png')
    plt.clf()
    # plt.show()
    corr_coef.append(np.corrcoef(np.vstack((all_ang,this_speed)))[0,1])

    # max error v.s. mismatch angle    
    poly = np.poly1d(np.polyfit(all_ang,this_error,deg=1))
    fit=poly([all_ang[0],all_ang[-1]])
    plt.scatter(all_ang,this_error)
    plt.plot([all_ang[0],all_ang[-1]],fit,'r',label='regression line')
    plt.xlabel("Mismatch Ang (deg)")
    plt.ylabel("Max Error (mm)")
    plt.legend()
    plt.title('Mismatch Ang v.s. Max Error, Cmd Speed:'+str(speed))
    # plt.show()
    plt.savefig(img_dir+"ang_error_speed_"+str(speed)+'.png')
    plt.clf()
    i+=1



plt.plot(all_speed,corr_coef,'o-')
plt.xlabel("Command Speed (mm/sec)")
plt.title("Correlation Coefficient of Speed at Max Error and Mismatch Angle")
plt.savefig(img_dir+'corrcoef_errspeed_ang.png')
# plt.show()
plt.clf()

# for combining result imgs
def trim_img(this_img):
    # trim from top
    i=0
    while True:
        if not np.all(this_img[i,:,:]==1.):
            break
        i += 1
    this_img = this_img[i:,:]
    # trim from bottom
    i=len(this_img)-1
    while True:
        if not np.all(this_img[i,:,:]==1.):
            break
        i -= 1
    this_img = this_img[:i,:]
    # trim from left
    i=0
    while True:
        if not np.all(this_img[:,i,:]==1.):
            break
        i += 1
    this_img = this_img[:,i:]
    # trim from right
    i=len(this_img[0])-1
    while True:
        if not np.all(this_img[:,i,:]==1.):
            break
        i -= 1
    this_img = this_img[:,:i]
    this_img = (this_img*255).astype(np.uint8)
    return this_img

# error and speed
the_img=np.array([])
for ang in all_ang:
    this_img=image.imread(img_dir+"error_speed_ang_"+str(ang)+'.png')
    this_img = trim_img(this_img)

    if len(the_img)==0:
        the_img=deepcopy(this_img)
    else:
        the_img=np.append(the_img,this_img,axis=1)
im = Image.fromarray(the_img)
im.save(img_dir+'error_speed_ang.png')

# angle and error
split=[0,6,12,len(all_speed)-1]
the_final_img = np.array([])
for split_i in range(len(split)-1):
    the_img=np.array([])
    for speed in all_speed[split[split_i]:split[split_i+1]]:
        this_img=image.imread(img_dir+"ang_error_speed_"+str(speed)+'.png')
        this_img = trim_img(this_img)

        if len(the_img)==0:
            the_img=deepcopy(this_img)
        else:
            the_img=np.append(the_img,this_img,axis=1)
    
    if len(the_final_img)==0:
        the_final_img=deepcopy(the_img)
    else:
        blank_img = np.zeros(the_final_img.shape).astype(np.uint8)
        blank_img[:len(the_img),:len(the_img[0]),:len(the_img[0,0])]=the_img
        the_final_img=np.append(the_final_img,blank_img,axis=0)
im = Image.fromarray(the_final_img)
im.save(img_dir+'ang_error_speed.png')

# angle and error
split=[0,6,12,len(all_speed)-1]
the_final_img = np.array([])
for split_i in range(len(split)-1):
    the_img=np.array([])
    for speed in all_speed[split[split_i]:split[split_i+1]]:
        this_img=image.imread(img_dir+"errspeed_ang_speed_"+str(speed)+'.png')
        this_img = trim_img(this_img)

        if len(the_img)==0:
            the_img=deepcopy(this_img)
        else:
            the_img=np.append(the_img,this_img,axis=1)
    
    if len(the_final_img)==0:
        the_final_img=deepcopy(the_img)
    else:
        blank_img = np.zeros(the_final_img.shape).astype(np.uint8)
        blank_img[:len(the_img),:len(the_img[0]),:len(the_img[0,0])]=the_img
        the_final_img=np.append(the_final_img,blank_img,axis=0)
im = Image.fromarray(the_final_img)
im.save(img_dir+'errspeed_ang_speed.png')
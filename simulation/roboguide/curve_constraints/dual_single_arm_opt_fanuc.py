import sys
import numpy as np
from pandas import *
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
import general_robotics_toolbox as rox
sys.path.append('../../../constraint_solver')
from constraint_solver import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
from error_check import *
from utils import *

data_type='blade'

if data_type=='blade':
    curve_data_dir='../../../data/from_NX/'
    data_dir='../data/curve_blade/'
elif data_type=='wood':
    curve_data_dir='../../../data/wood/'
    data_dir='../data/curve_wood/'

# robot2_type='dual_single_arm_freeze' # robot2 not moving
# robot2_type='dual_single_arm_straight/' # robot2 is multiple user defined straight line
# robot2_type='dual_single_arm_straight_min/' # robot2 is multiple user defined straight line
robot2_type='dual_single_arm_straight_min10/' # robot2 is multiple user defined straight line
Path(data_dir+robot2_type).mkdir(exist_ok=True)

# read curve relative path
relative_path=read_csv(curve_data_dir+"Curve_dense.csv",header=None).values
# relative_path=relative_path[::100] # downsample for double check

# load relative position of robot 2
with open(data_dir+'m900ia.yaml') as file:
    H_robot2 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
base2_R=H_robot2[:3,:3]
base2_p=1000*H_robot2[:-1,-1]

# robot1, holding spray gun
robot1=m710ic(d=50)

# robot2, carring the curve
with open(data_dir+'tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
robot2=m900ia(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

#read in initial curve pose
with open(data_dir+'blade_pose.yaml') as file:
    blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

###### get tcp
# import general_robotics_toolbox as rox
# bT = rox.Transform(blade_pose[:3,:3],blade_pose[:3,3])
# bT=rox.Transform(R=rox.rot([0,0,1],np.pi),p=[3500,0,0])*bT
# # print(bT)
# # print(R2wpr(bT.R))
# # exit()
# robot=m900ia(R_tool=np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),p_tool=np.array([0,0,0])*1000.,d=0)
# # T_tool=robot.fwd(np.deg2rad([0,49.8,-17.2,0,65.4,0]))
# T_tool=robot.fwd(np.deg2rad([0,48.3,-7,0,55.8,0]))
# # T_tool=robot.fwd(np.deg2rad([0,0,0,0,0,0]))
# print(T_tool)
# bT=T_tool.inv()*bT
# bT=rox.Transform(np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),np.array([0,0,0])*1000.)*bT
# # print(bT)
# # print(R2wpr(bT.R))
# # exit()
# bT_T=np.vstack((np.vstack((bT.R.T,bT.p)).T,[0,0,0,1]))
# # print(bT_T)
# with open(data_dir+'dual_arm/tcp.yaml','w') as file:
#     yaml.dump({'H':bT_T.tolist()},file)
# with open(data_dir+'dual_arm/tcp.yaml') as file:
#     H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
# # print(H_tcp)
# # bT_array=
# # print(bT.inv())
# from fanuc_motion_program_exec_client import *
# print(R2wpr(bT.R))
# exit()
##################################################

curve_js1=read_csv(data_dir+"Curve_js.csv",header=None).values

opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,steps=50000)

# ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)
ms = MotionSendFANUC(robot1=robot1,robot2=robot2)
q_init1=curve_js1[0]
q_init2=ms.calc_robot2_q_from_blade_pose(blade_pose,base2_R,base2_p)
print(np.rad2deg(q_init2))

##### user defined robot2 curve
## 1. robot2 does not move
q_out2=np.tile(q_init2,(len(relative_path),1)) 
## 2. robot2 move to defined pose with defined steps
# robot2_path=np.array([q_init2,np.deg2rad([9.1,39.2,-19.3,-18.4,37.4,13.4]),np.deg2rad([17.9,42.4,-13.5,-44.4,27.9,33.5]),\
#                     np.deg2rad([24.6,51.9,17,-147.9,38.5,133.8])])
# robot2_step=[12500,12500,24999]
# robot2_path=np.array([q_init2,np.deg2rad([11.5,32.8,-26.6,-34.6,25.4,27.9]),np.deg2rad([18.7,38.4,-8.2,-114.0,20.4,103.3])])
# robot2_step=[12500,37499]
# robot2_path=np.array([q_init2,np.deg2rad([18.0,52.8,-10.9,-20.2,63.5,8.8])])
# robot2_step=[49999]
# robot2_path=np.array([q_init2,np.deg2rad([22.4,59.6,2.9,-28.3,52.2,15.6])])
# robot2_step=[49999]
# robot2_path=np.array([q_init2,np.deg2rad([2.9,53,-10.4,-3.3,61.9,1.5])]) # min (100)
# robot2_step=[49999]
robot2_path=np.array([q_init2,np.deg2rad([0.3,50.2,-16.5,-0.4,65.1,0.2])]) # min10
robot2_step=[49999]
###########################################

q_out2=[q_init2]
assert np.sum(robot2_step) == (len(relative_path)-1),print("must equal to relative path length")
for i in range(1,len(robot2_path)):
    print("target:",i,"steps:",robot2_step[i-1])
    pose_start=robot2.fwd(robot2_path[i-1])
    p_start=pose_start.p
    R_start=pose_start.R
    pose_end=robot2.fwd(robot2_path[i])
    p_end=pose_end.p
    R_end=pose_end.R
    #find slope
    slope_p=p_end-p_start
    slope_p=slope_p/np.linalg.norm(slope_p)
    #find k,theta
    k,theta=rox.R2rot(R_end@R_start.T)

    # adding extension with uniform space
    extend_step_d=np.linalg.norm(p_end-p_start)/robot2_step[i-1]
    for j in range(1,robot2_step[i-1]+1):
        p_extend=p_start+j*extend_step_d*slope_p
        theta_extend=np.linalg.norm(p_extend-p_start)*theta/np.linalg.norm(p_end-p_start)
        R_extend=rox.rot(k,theta_extend)@R_start
        # ik
        q_out2.append(car2js(robot2,q_out2[-1],p_extend,R_extend)[0])
###############################################################
q_out2=np.array(q_out2)
df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
df.to_csv(data_dir+robot2_type+'arm2.csv',header=False,index=False)

#convert curve to base frame
base2_T=rox.Transform(base2_R,base2_p)
curve_base1=[]
curve_normal_base1=[]
for i in range(len(q_out2)):
    T_r1_r2tool = base2_T*robot2.fwd(q_out2[i]) # Transform from r1 base to r2 tool
    this_curve_base1= T_r1_r2tool.p + np.matmul(T_r1_r2tool.R, relative_path[i,:3])
    this_curve_normal_base1=np.matmul(T_r1_r2tool.R,relative_path[i,3:])
    curve_base1.append(this_curve_base1)
    curve_normal_base1.append(this_curve_normal_base1)
curve_base1=np.array(curve_base1)
curve_normal_base1=np.array(curve_normal_base1)

q_out1=opt.single_arm_stepwise_optimize(q_init1,curve=curve_base1,curve_normal=curve_normal_base1)

####output to trajectory csv
df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
df.to_csv(data_dir+robot2_type+'arm1.csv',header=False,index=False)


##########path verification####################
relative_path_out,relative_path_out_R=ms.form_relative_path(q_out1,q_out2,base2_R,base2_p)
# error,angle_error=calc_all_error_w_normal(relative_path_out,relative_path[:,:3],relative_path_out_R[:,:,-1],relative_path[:,3:])
error=np.linalg.norm(relative_path[:,:3]-relative_path_out,axis=1)
plt.plot(error,label='Euclidean Error')
# plt.plot(angle_error,label='Angular Error')
plt.legend()
plt.title('error plot')
plt.show()

dlam=np.linalg.norm(np.diff(relative_path_out,axis=0),2,1)
plt.plot(dlam)
plt.title('dlam')
plt.show()

###dual lambda_dot calc
dlam=calc_lamdot_2arm(np.hstack((q_out1,q_out2)),opt.lam,ms.robot1,ms.robot2,step=1)
print('lamdadot min: ', min(dlam))

plt.plot(opt.lam,dlam,label="lambda_dot_max")
plt.xlabel("lambda")
plt.ylabel("lambda_dot")
plt.title("DUALARM max lambda_dot vs lambda (path index)")
plt.ylim([0,3500])
# plt.savefig(data_dir+"results.png")
plt.show()

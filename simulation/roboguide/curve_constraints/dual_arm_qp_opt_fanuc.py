import sys
import numpy as np
from pandas import *
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
sys.path.append('../../../constraint_solver')
from constraint_solver import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *

# data_type='blade_shift'
data_type='blade'

if data_type=='blade':
    curve_data_dir='../../../data/from_NX/'
    data_dir='../data/curve_blade/'
elif data_type=='wood':
    curve_data_dir='../../../data/wood/'
    data_dir='../data/curve_wood/'
elif data_type=='blade_shift':
    curve_data_dir='../../../data/blade_shift/'
    data_dir='../data/curve_blade_shift/'

robot2_type='dual_arm_qp/'
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
# T_tool=robot.fwd(np.deg2rad([0,49.8,-17.2,0,65.4,0])) # blade
# # T_tool=robot.fwd(np.deg2rad([0,48.3,-7,0,55.8,0])) # wood
# # T_tool=robot.fwd(np.deg2rad([0,0,0,0,0,0]))
# # print(T_tool)
# bT=T_tool.inv()*bT
# bT=rox.Transform(np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),np.array([0,0,0])*1000.)*bT
# # print(bT)
# # print(R2wpr(bT.R))
# # exit()
# bT_T=np.vstack((np.vstack((bT.R.T,bT.p)).T,[0,0,0,1]))
# # print(bT_T)
# with open(data_dir+'tcp.yaml','w') as file:
#     yaml.dump({'H':bT_T.tolist()},file)
# with open(data_dir+'tcp.yaml') as file:
#     H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
# # print(H_tcp)
# # bT_array=
# # print(bT.inv())
# from fanuc_motion_program_exec_client import *
# bT=rox.Transform(np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),np.array([0,0,0])*1000.)*bT
# print(bT)
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

ldotdes = 1000 ## 1000 mm/sec

KR = 1
Kp = 1
# Kq = np.append([0.1,0.1,0.1,0.2,0.2,0.2],[0.2,0.2,0.2,0.4,0.4,0.4])
Kq = np.ones(12)*0.01

q_out1, q_out2=opt.dual_arm_qp_vel_track(q_init1,q_init2,ldotdes,Kq,KR,Kp)
# q_out1 = np.array(read_csv(data_dir+robot2_type+'arm1.csv',header=None).values)
# q_out2 = np.array(read_csv(data_dir+robot2_type+'arm2.csv',header=None).values)
##########path verification####################
relative_path_out,relative_path_out_R=ms.form_relative_path(q_out1,q_out2,base2_R,base2_p)
plt.plot(np.linalg.norm(relative_path[:,:3]-relative_path_out,axis=1))
plt.title('error plot')
plt.show()

dlam=np.linalg.norm(np.diff(relative_path_out,axis=0),2,1)
plt.plot(dlam)
plt.title('dlam')
plt.show()

####output to trajectory csv
df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
df.to_csv(data_dir+robot2_type+'arm1.csv',header=False,index=False)
df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
df.to_csv(data_dir+robot2_type+'arm2.csv',header=False,index=False)

###dual lambda_dot calc
dlam=calc_lamdot_2arm(np.hstack((q_out1,q_out2)),opt.lam,ms.robot1,ms.robot2,step=1)
print('lamdadot min: ', min(dlam))

plt.plot(opt.lam,dlam,label="lambda_dot_max")
plt.xlabel("lambda")
plt.ylabel("lambda_dot")
plt.title("DUALARM max lambda_dot vs lambda (path index)")
plt.ylim([0,3500])
# plt.savefig("trajectory/results.png")
plt.show()

ax = plt.axes(projection='3d')
ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
ax.plot3D(relative_path_out[:,0], relative_path_out[:,1],relative_path_out[:,2], 'green',label='qp')
ax.view_init(61, -67)
plt.legend()
# plt.savefig(ilc_output+'traj_iteration_'+str(i))
# plt.clf()
plt.show()

########################## plot joint ##########################
ldot_des = 1000
lam = np.append(0,np.cumsum(np.linalg.norm(np.diff(relative_path_out,axis=0),2,1)))
dt = np.linalg.norm(np.diff(relative_path_out,axis=0),2,1)/1000
timestamp = np.append(0,np.cumsum(dt))

plot_joint=True
if plot_joint:
    q_out1=np.array(q_out1)
    q_out2=np.array(q_out2)
    for i in range(1,3):
        # robot
        if i==1:
            this_robot=robot1
            this_curve_js_exe=q_out1
        else:
            this_robot=robot2
            this_curve_js_exe=q_out2
        fig, ax = plt.subplots(4,6)
        dt=np.diff(timestamp)
        dt=np.append(dt[0],dt)
        for j in range(6):
            ax[0,j].plot(timestamp,this_curve_js_exe[:,j])
            ax[0,j].axis(ymin=this_robot.lower_limit[j]*1.05,ymax=this_robot.upper_limit[j]*1.05)
            # dq=np.gradient(this_curve_js_exe[:,j])
            dq=np.diff(this_curve_js_exe[:,j])
            dq=np.append(dq[0],dq)
            dqdt=dq/dt
            ax[1,j].plot(timestamp,dqdt)
            ax[1,j].axis(ymin=-this_robot.joint_vel_limit[j]*1.05,ymax=this_robot.joint_vel_limit[j]*1.05)
            # d2qdt2=np.gradient(dqdt)/dt
            ddqdt = np.diff(dqdt)
            ddqdt=np.append(ddqdt[0],ddqdt)
            d2qdt2=ddqdt/dt
            ax[2,j].plot(timestamp,d2qdt2)
            ax[2,j].axis(ymin=-this_robot.joint_acc_limit[j]*1.05,ymax=this_robot.joint_acc_limit[j]*1.05)
            # d3qdt3=np.gradient(d2qdt2)/dt
            d3qdt2=np.diff(d2qdt2)
            d3qdt2=np.append(d3qdt2[0],d3qdt2)
            d3qdt3=np.diff(d2qdt2)/dt
            ax[3,j].plot(timestamp,d3qdt3)
            ax[3,j].axis(ymin=-this_robot.joint_jrk_limit[j]*1.05,ymax=this_robot.joint_jrk_limit[j]*1.05)
        # plt.title('Robot '+str(i)+' joint trajectoy/velocity/acceleration.')
        plt.show()

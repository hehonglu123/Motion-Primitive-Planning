import sys
sys.path.append('../')
from constraint_solver import *

from MotionSend import *

data_dir='../../data/from_NX/'
relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

with open(data_dir+'dual_arm/abb1200.yaml') as file:
    H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

base2_R=H_1200[:3,:3]
base2_p=1000*H_1200[:-1,-1]

with open(data_dir+'dual_arm/tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)

#read in initial curve pose
with open(data_dir+'blade_pose.yaml') as file:
	blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

curve_js1=read_csv(data_dir+"Curve_js.csv",header=None).values


opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=ms.robot1,robot2=ms.robot2,base2_R=base2_R,base2_p=base2_p,steps=50000)


q_init1=curve_js1[0]
q_init2=ms.calc_robot2_q_from_blade_pose(blade_pose,base2_R,base2_p)

q_out1, q_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2)
##########path verification####################
ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)
relative_path_out,relative_path_out_R=ms.form_relative_path(q_out1,q_out2,base2_R,base2_p)
plt.plot(np.linalg.norm(relative_path[:,:3]-relative_path_out,axis=1))
plt.title('error plot')
plt.show()

####output to trajectory csv
df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
df.to_csv('trajectory/'+'arm1.csv',header=False,index=False)
df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
df.to_csv('trajectory/'+'arm2.csv',header=False,index=False)

###dual lambda_dot calc
dlam=calc_lamdot_2arm(np.hstack((q_out1,q_out2)),opt.lam,ms.robot1,ms.robot2,step=1)
print('lamdadot min: ', min(dlam))

plt.plot(opt.lam,dlam,label="lambda_dot_max")
plt.xlabel("lambda")
plt.ylabel("lambda_dot")
plt.title("DUALARM max lambda_dot vs lambda (path index)")
plt.ylim([0,3500])
plt.savefig("trajectory/results.png")
plt.show()

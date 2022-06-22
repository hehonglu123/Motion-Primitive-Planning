import sys
sys.path.append('../')
from constraint_solver import *

data_dir='../../data/wood/'
relative_path=read_csv(data_dir+"relative_path_tool_frame.csv",header=None).values

robot1=abb1200(d=50)
robot2=abb6640()
base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
base2_p=np.array([3000,1000,0])
opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,steps=50000)


q_init1=np.array([1.001043036,	0.117102871,	0.238506285,	0.888767966,	-1.224797607,	-1.039746248])
q_init2=np.array([0.124940222,	0.621755233,	0.461322728,	-0.32124382,	-1.58550511,	0.119284092])

q_out1, q_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2)

####output to trajectory csv
df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
df.to_csv('trajectory/'+'arm1.csv',header=False,index=False)
df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
df.to_csv('trajectory/'+'arm2.csv',header=False,index=False)

###dual lambda_dot calc
dlam=calc_lamdot_2arm(np.hstack((q_out1,q_out2)),opt.lam,robot1,robot2,step=1)
print('lamdadot min: ', min(dlam))

plt.plot(opt.lam,dlam,label="lambda_dot_max")
plt.xlabel("lambda")
plt.ylabel("lambda_dot")
plt.title("DUALARM max lambda_dot vs lambda (path index)")
plt.ylim([0,3500])
plt.savefig("trajectory/results.png")
plt.show()

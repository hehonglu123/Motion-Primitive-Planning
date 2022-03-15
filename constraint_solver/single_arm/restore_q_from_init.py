import sys, yaml
sys.path.append('../')
from constraint_solver import *

col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

theta0=2.07499

robot=abb6640(d=50)
opt=lambda_opt(curve,curve_normal,robot1=robot)

R_temp=opt.direction2R(opt.curve_normal[0],-curve[1]+curve[0])
R=np.dot(R_temp,Rz(theta0))
q_init=opt.robot1.inv(curve[0],R)[0]
q_out=opt.single_arm_stepwise_optimize(q_init,curve,curve_normal)



###output to csv
df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
df.to_csv('trajectory/init_opt/init_opt_js.csv',header=False,index=False)

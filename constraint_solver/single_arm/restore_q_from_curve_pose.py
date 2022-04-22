import sys, yaml
sys.path.append('../')
from constraint_solver import *

###read actual curve
col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("trajectory/curve_pose_opt/arm1.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

###read actual curve
col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("../../data/from_ge/relative_path.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

with open('trajectory/curve_pose_opt/curve_pose.yaml') as file:
	curve_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)


robot=abb6640(d=50)
opt=lambda_opt(curve,curve_normal,robot1=robot)

R_curve=curve_pose[:3,:3]
shift=curve_pose[:-1,-1]*1000

curve_new=np.dot(R_curve,curve.T).T+np.tile(shift,(len(curve),1))
curve_normal_new=np.dot(R_curve,curve_normal.T).T


q_out=opt.single_arm_stepwise_optimize(curve_js[0],curve_new,curve_normal_new)
###output to csv
df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
df.to_csv('trajectory/single_arm/curve_pose_opt/curve_pose_opt_js.csv',header=False,index=False)

df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1], 'z':curve_new[:,2],'x_direction':curve_normal_new[:,0],'y_direction':curve_normal_new[:,1],'z_direction':curve_normal_new[:,2]})
df.to_csv('trajectory/single_arm/curve_pose_opt/curve_pose_opt_cs.csv',header=False,index=False)
import sys, yaml
sys.path.append('../')
from constraint_solver import *


robot=abb6640(d=50)

###read actual curve
col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("trajectory/all_theta_opt/arm1.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

###read actual curve
# col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
# data = read_csv("../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
# curve_x=data['X'].tolist()
# curve_y=data['Y'].tolist()
# curve_z=data['Z'].tolist()
# curve_direction_x=data['direction_x'].tolist()
# curve_direction_y=data['direction_y'].tolist()
# curve_direction_z=data['direction_z'].tolist()
# curve=np.vstack((curve_x, curve_y, curve_z)).T
# curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

data = read_csv("../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.1/curve_fit.csv")
curve_x=data['x'].tolist()
curve_y=data['y'].tolist()
curve_z=data['z'].tolist()
curve_direction_x=data['R3'].tolist()
curve_direction_y=data['R6'].tolist()
curve_direction_z=data['R9'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

###get breakpoints
data = read_csv('../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.1/command.csv')
breakpoints=np.array(data['breakpoints'].tolist())
breakpoints[1:]=breakpoints[1:]-1

opt=lambda_opt(curve,curve_normal,robot1=abb6640(d=50),idx=breakpoints)

opt=lambda_opt(curve,curve_normal,robot1=robot)
x=np.array([ 2.37981974, -1.77127285, -1.51123952, -1.03392819, -0.98898889,
       -1.46282972, -1.73447395, -1.12511445, -0.70919136, -0.70038516,
       -0.65993072, -0.60959948, -0.66672792, -0.11653925, -0.84033196,
       -0.53800397, -0.89930259, -2.22511634, -1.63561632, -1.68750846,
       -2.13820504])
num_per_step=int(len(curve)/opt.steps)	

theta_all=[]
for i in range(len(x)-1):
	theta_all=np.append(theta_all,np.linspace(x[i],x[i+1],(opt.idx[i+1]-opt.idx[i])))
theta_all=np.append(theta_all,x[-1]*np.ones(len(curve)-len(theta_all)))

for i in range(len(theta_all)):
	print(i)
	if i==0:
		R_temp=opt.direction2R(curve_normal[0],-curve[1]+curve[0])
		R=np.dot(R_temp,Rz(theta_all[i]))
		q_out=[opt.robot1.inv(curve[i],R)[0]]

	else:
		R_temp=opt.direction2R(curve_normal[i],-curve[i]+curve[i-1])
		R=np.dot(R_temp,Rz(theta_all[i]))
		###get closet config to previous one
		q_inv_all=opt.robot1.inv(curve[i],R)
		temp_q=q_inv_all-q_out[-1]
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		q_out.append(q_inv_all[order[0]])

q_out=np.array(q_out)

###output to csv
df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
df.to_csv('trajectory/all_theta_opt_blended/all_theta_opt_js.csv',header=False,index=False)

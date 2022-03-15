import sys, yaml
sys.path.append('../')
from constraint_solver import *

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


robot=abb6640(d=50)
opt=lambda_opt(curve,curve_normal,robot1=robot)
x=np.array([ 0.00099455,  0.04053725,  0.0597389 ,  0.08903915,  0.06229287,
        0.02856117, -0.07002132, -0.1024818 , -0.01162368, -0.00299374,
       -0.0683647 , -0.14392112, -0.07377098, -0.06561495, -0.10404596,
       -0.0735651 , -0.00669726,  0.06090607,  0.00289383,  0.02550258,
        0.06721874,  0.06377314,  0.01631902, -0.03542485, -0.05970686,
       -0.04923485, -0.0193444 ,  0.01146988,  0.01435165, -0.01696077,
       -0.04725104,  0.07471899,  0.15876755])
num_per_step=int(len(curve)/opt.steps)	

theta_all=[]
for i in range(len(x)-1):
	theta_all=np.append(theta_all,np.linspace(x[i],x[i+1],num_per_step))
theta_all=np.append(theta_all,x[-1]*np.ones(len(curve)-len(theta_all)))

for i in range(len(theta_all)):
	print(i)
	if i==0:
		R_temp=opt.direction2R(curve_normal[i],-curve[i+1]+curve[i])
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
df.to_csv('trajectory/all_theta_opt/all_theta_opt_js.csv',header=False,index=False)

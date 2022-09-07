import sys
import numpy as np
import yaml
from pandas import *
from matplotlib import pyplot as plt
import general_robotics_toolbox as rox
sys.path.append('../../../constraint_solver')
from constraint_solver import *

# data_type='wood'
data_type='blade'

if data_type=='blade':
    curve_data_dir='../../../data/from_NX/'
    data_dir='../data/curve_blade/'
elif data_type=='wood':
    curve_data_dir='../../../data/wood/'
    data_dir='../data/curve_wood/'

# read curve relative path
relative_path=read_csv(curve_data_dir+"Curve_dense.csv",header=None).values
# read curve pose from baseline
with open(data_dir+'blade_pose.yaml') as file:
    blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
relative_path_p = np.hstack((relative_path[:,:3],np.ones((len(relative_path),1))))
relative_path_n = relative_path[:,3:]

## curve frame conversion
curve_p = np.matmul(blade_pose,relative_path_p.T)[:-1,:]
curve_p = curve_p.T
curve_n = np.matmul(blade_pose[:3,:3],relative_path_n.T).T
curve = np.hstack((curve_p,curve_n))
## baseline js
curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values

robot=m710ic(d=50)
opt=lambda_opt(curve[:,:3],curve[:,3:],robot1=robot,steps=50000)
q_init=curve_js[0]

q_out=opt.single_arm_stepwise_optimize(q_init,curve=curve[:,:3],curve_normal=curve[:,3:])

####output to trajectory csv
df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
df.to_csv(data_dir+'single_arm/arm1.csv',header=False,index=False)

dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)

plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
plt.xlabel("lambda")
plt.ylabel("lambda_dot")
plt.ylim([0,2000])
plt.title("max lambda_dot vs lambda (path index)")
plt.show()
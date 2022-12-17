import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from scipy.optimize import minimize, differential_evolution


sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *

def find_H(x,curve,curve_exe):
	theta=np.linalg.norm(x[:3])
	k=x[:3]/theta
	H=H_from_RT(rot(k,theta),x[3:])

	curve_rotated=[]
	for i in range(len(curve_exe)):
		curve_rotated.append(np.dot(H,np.hstack((curve_exe[i],[1])).T)[:-1])

	out=np.linalg.norm(calc_all_error(np.array(curve_rotated),curve))
	return out



dataset='curve_1'
curve = read_csv("../data/"+dataset+'/baseline/Curve_in_base_frame.csv',header=None).values


data = np.loadtxt(dataset+'_base_frame/end_waypoints.csv',delimiter=',', skiprows=1,usecols = (0,1,2,3,4,5,6))

curve_exe=data[:,1:4]*1000
error=calc_all_error(curve_exe,curve[:,:3])

lam=calc_lam_cs(curve_exe)

res = minimize(find_H, [1,0,0,0,0,0], method='SLSQP', args=(curve[:,:3],curve_exe,),options={'maxiter': 1000, 'disp': True}) 


# lowerer_limit=np.array([-3000,-3000,-3000,-2*np.pi,-2*np.pi,-2*np.pi])
# upper_limit=np.array([3000,3000,3000,2*np.pi,2*np.pi,2*np.pi])
# bnds=tuple(zip(lowerer_limit,upper_limit))

# res = differential_evolution(find_H, bnds,args=(curve[:,:3],curve_exe),workers=-1,
# 									x0 = [0,0,0,1,0,0],
# 									strategy='best1bin', maxiter=10,
# 									popsize=15, tol=1e-10,
# 									mutation=(0.5, 1), recombination=0.7,
# 									seed=None, callback=None, disp=True,
# 									polish=True, init='latinhypercube',
# 									atol=0.)
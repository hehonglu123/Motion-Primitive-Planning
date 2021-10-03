import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from toolbox_circular_fit import *
from scipy.optimize import differential_evolution
import traceback

def DE_stepwise_3dfitting(breakpoints,curve):
	breakpoints=np.sort(np.unique(breakpoints.astype(int)))
	breakpoints=np.append(breakpoints,len(curve)-1)
	breakpoints=np.insert(breakpoints,0,0)
	
	fit=stepwise_3dfitting(curve,breakpoints)
	print(breakpoints)
	error=[]
	for i in range(len(fit)):
		error_temp=np.linalg.norm(curve-fit[i],axis=1)
		idx=np.argmin(error_temp)
		error.append(error_temp[idx])

	try:
		return np.max(np.array(error))
	except:
		traceback.print_exc()

def fit_w_breakpoints3d(curve,num_breakpoints):
	bounds = np.zeros([num_breakpoints-2, 2])
	bounds[:, 0] = 1
	bounds[:, 1] = len(curve)-2

	temp_num=int(len(curve)/(num_breakpoints-1))

	res = differential_evolution(DE_stepwise_3dfitting, bounds, args=(curve,),workers=-1,
									# x0 = np.arange(0,len(curve-1),temp_num),
									strategy='best1bin', maxiter=20,
									popsize=10*(num_breakpoints-2), tol=0.001,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=1.)

	print(DE_stepwise_3dfitting(res.x,curve))

def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	fit_w_breakpoints3d(curve,7) ###max20 iter error@ 1.58

if __name__ == "__main__":
	main()

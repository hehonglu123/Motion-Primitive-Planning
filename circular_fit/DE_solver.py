import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from toolbox_circular_fit import *
from scipy.optimize import differential_evolution



def fit_w_breakpoints3d(curve,num_breakpoints):
	bounds = np.zeros([num_breakpoints, 2])
	bounds[:, 0] = 0
	bounds[:, 1] = len(curve)-1

	res = differential_evolution(DE_stepwise_3dfitting, bounds,
									args=(curve,))#,
									# strategy='best1bin', maxiter=1000,
									# popsize=50, tol=1e-3,
									# mutation=(0.5, 1), recombination=0.7,
									# seed=None, callback=None, disp=False,
									# polish=True, init='latinhypercube',
									# atol=1e-4)


def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	fit_w_breakpoints3d(curve,2)

if __name__ == "__main__":
	main()

from pandas import *
import numpy as np
import sys


def find_j_min(robot,curve_js):
	sing_min=[]
	for q in curve_js:
		u, s, vh = np.linalg.svd(robot.jacobian(q))
		sing_min.append(s[-1])

	return np.min(sing_min),np.argmin(sing_min)

from pandas import *
import numpy as np
import sys
from robot_def import *



def find_j_min(curve_js):
	sing_min=[]
	for q in curve_js:
		u, s, vh = np.linalg.svd(jacobian(q))
		sing_min.append(s[-1])

return np.min(sing_min),np.argmin(sing_min)

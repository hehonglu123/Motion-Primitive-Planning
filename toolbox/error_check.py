import numpy as np
import copy

###calculate distance between point to line
def get_distance(p1,p2,p3):
	return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

###calculate maximum error of 1 point and the curve in cartesian space, distance only
# def calc_error(p,curve):
# 	dist=np.linalg.norm(curve-np.tile(p,(len(curve),1)),axis=1)
# 	order=np.argsort(dist)
# 	error=get_distance(curve[order[0]],curve[order[1]],p)
# 	return error
def calc_error(p,curve):
	dist=np.linalg.norm(curve-np.tile(p,(len(curve),1)),axis=1)
	order=np.argsort(dist)
	return dist[order[0]]
	
###calculate maximum error between fit curve and original curve in cartesian space, distance only
def calc_max_error(fit,curve):
	max_error=0
	for p in fit:
		error=calc_error(p,curve)
		if error>max_error:
			max_error=copy.deepcopy(error)
	return max_error

def calc_avg_error(fit,curve):
	error=0
	for p in fit:
		error+=calc_error(p,curve)
	return error/len(fit)
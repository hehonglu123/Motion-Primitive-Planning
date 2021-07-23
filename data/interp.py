import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
import sys
import numpy as np

def main():
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("Curve.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	col_names=['X', 'Y', 'Z'] 
	data = read_csv("Curve_interp.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_interp=np.vstack((curve_x, curve_y, curve_z)).T

	curve_direction_interp=np.zeros(curve_interp.shape)

	###orientation interpolation
	for i in range(len(curve_interp)):
		temp=curve-curve_interp[i]
		order=np.argsort(np.linalg.norm(temp,axis=1))
		curve_segment_distance=np.linalg.norm(curve[order[0]]-curve[order[1]])
		curve_direction_interp[i,:]=curve_direction[order[0]]*np.linalg.norm(curve_interp[i]-curve[order[1]])/curve_segment_distance+curve_direction[order[1]]*np.linalg.norm(curve_interp[i]-curve[order[0]])/curve_segment_distance
	###output to csv
	df=DataFrame({'x':curve_interp[:,0],'y':curve_interp[:,1], 'z':curve_interp[:,2],'direction_x':curve_direction_interp[:,0],'direction_y':curve_direction_interp[:,1], 'direction_z':curve_direction_interp[:,2]})
	df.to_csv('Curve_interp.csv',header=False,index=False)

if __name__ == "__main__":
	main()
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *
from error_check import *

def main():
	col_names=['timestamp', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data = read_csv("comparison/moveL+moveC/threshold05/v50_z10.csv",names=col_names)
	data = data.apply(to_numeric, errors='coerce')
	q1=data['J1'].tolist()[1:-1]
	q2=data['J2'].tolist()[1:-1]
	q3=data['J3'].tolist()[1:-1]
	q4=data['J4'].tolist()[1:-1]
	q5=data['J5'].tolist()[1:-1]
	q6=data['J6'].tolist()[1:-1]
	timestamp=data['timestamp'].tolist()[1:-1]
	q_all=np.vstack((q1,q2,q3,q4,q5,q6)).T

	speed=[]
	curve=[]
	for i in range(len(q_all)):
		q=np.radians(q_all[i])
		curve.append(fwd(q).p)
		if len(curve)>1:
			speed.append(np.linalg.norm(curve[-1]-curve[-2])/(timestamp[i]-timestamp[i-1]))

	speed=np.array(speed)
	print(np.min(speed[np.nonzero(speed)]),np.max(speed),np.average(speed))

if __name__ == "__main__":
	main()
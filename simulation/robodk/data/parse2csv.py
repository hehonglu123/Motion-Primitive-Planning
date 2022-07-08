import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../../../toolbox')
from robots_def import *


def main():
	robot=abb6640(d=50)
	primitives=[]
	q_bp=[]
	p_bp=[]

	dataset='wood/'


	f = open(dataset+'dk_program', "r")
	for line in f.readlines():
		if 'r.Move' == line[:6]:
			if 'J'== line[6]:
				primitives.append('movej_fit')
				q=line[14:-8].split(',')
				q=np.radians(list(map(float, q)))
				q_bp.append([q])
				p_bp.append([robot.fwd(q).p])

			if 'L'== line[6]:
				primitives.append('movel_fit')

				q=line[10:-11].split('[')[-1].split(',')

				q=np.radians(list(map(float, q)))
				q_bp.append([q])
				p_bp.append([robot.fwd(q).p])

	breakpoints=np.zeros(len(q_bp))
	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'points':p_bp,'q_bp':q_bp})
	df.to_csv(dataset+'command.csv',header=True,index=False)

if __name__ == "__main__":
	main()
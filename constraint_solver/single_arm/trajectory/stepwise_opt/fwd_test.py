import sys
from pandas import *
sys.path.append('../../../../toolbox')
from robots_def import *

robot=abb6640(d=50)
curve_js = read_csv('arm1.csv',header=None).values
for i in range(len(curve_js)):
	print(i)
	robot.fwd(curve_js[i])
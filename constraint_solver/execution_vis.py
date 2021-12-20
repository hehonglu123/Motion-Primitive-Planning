from pandas import *
import numpy as np
import sys
sys.path.append('../toolbox')
from robot_def import *

def format_movej(q):

	eax='[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]'
	speed='v5000'
	zone='z10'
	q_deg=np.degrees(q)
	return 'MoveAbsJ '+'[['+str(q_deg[0])+','+str(q_deg[1])+','+str(q_deg[2])+','+str(q_deg[3])+','+str(q_deg[4])+','+str(q_deg[5])+'],'+eax+'],'\
			+speed+','+zone+',Paintgun;'

col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv('curve_poses/curve_pose5/Curve_backproj_js0.csv', names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T


for i in range(0,len(curve_js),1000):
	print(format_movej(curve_js[i]))
print(format_movej(curve_js[-1]))
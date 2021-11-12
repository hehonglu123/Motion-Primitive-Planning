from pandas import *
import numpy as np
import sys
sys.path.append('../toolbox')
from robot_def import *



col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("from_cad/Curve_backproj_js.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T


sing_min=[]
for q in curve_js:
	u, s, vh = np.linalg.svd(jacobian(q))
	sing_min.append(s[-1])

print(np.min(sing_min),np.argmin(sing_min))

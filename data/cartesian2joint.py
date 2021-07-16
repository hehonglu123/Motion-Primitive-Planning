import numpy as np
from pandas import *
import sys

sys.path.append('../toolbox')
from robot_def import *



col_names=['X', 'Y', 'Z'] 
data = read_csv("Curve_interp.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

curve_base=np.zeros(curve.shape)
curve_base[:,0]=2700+curve[:,2]
curve_base[:,1]=-800+curve[:,0]
curve_base[:,2]=500+curve[:,1]


curve_base=curve_base/1000.

curve_js=np.zeros((len(curve),6))

for i in range(len(curve_base)):
	curve_js[i]=inv(curve_base[i])


###output to csv
df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
df.to_csv('Curve_js.csv')
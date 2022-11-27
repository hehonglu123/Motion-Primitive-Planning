import numpy as np

curve_js2=np.loadtxt('arm2.csv',delimiter=',')
curve_js2[:,-1]=curve_js2[:,-1]-np.radians(30)
np.savetxt('arm2.csv',curve_js2,delimiter=',')
import numpy as np
import time, sys
from pandas import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
sys.path.append('../../../toolbox/egm_toolbox')
from EGM_toolbox import *
import rpi_abb_irc5

egm = rpi_abb_irc5.EGM()

dataset='wood/'
data_dir='../../../data/'
curve_js = read_csv(data_dir+dataset+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+dataset+'Curve_in_base_frame.csv',header=None).values
vd=200

robot=abb6640(d=50)
lam=calc_lam_cs(curve[:,:3])
ts=0.004

et=EGM_toolbox(egm,robot)

idx=et.downsample24ms(curve,vd)


###move to start first
print('moving to start point')
et.jog_joint(curve_js[0])


timestamp,curve_exe_js=et.traverse_curve_js(curve_js[idx])

plt.plot(timestamp)
plt.show()

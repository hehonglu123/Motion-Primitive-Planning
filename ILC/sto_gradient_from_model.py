########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
from ilc_toolbox import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	# data_dir="fitting_output_new/python_qp_movel/"
	dataset='from_NX/'
	data_dir="../data/"+dataset
	fitting_output="../data/"+dataset+'baseline/100L/'


	curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	max_error_threshold=0.1
	robot=abb6640(d=50)

	v=1100
	s = speeddata(v,9999999,9999999,999999)
	z = z10


	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')

	###extension
	p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	_,G=ilc.sto_gradient_from_model(p_bp,q_bp,100)

	im=plt.imshow(G, cmap='hot', interpolation='nearest')
	plt.colorbar(im)
	plt.title("Stochastic Gradient from Analytical Model")
	plt.xlabel('d_bp')
	plt.ylabel('d_p_model')
	plt.show()
if __name__ == "__main__":
	main()
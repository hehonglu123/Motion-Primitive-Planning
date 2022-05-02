from pandas import read_csv
from abb_motion_program_exec_client import *
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *

def main():
	quatR = R2q(rot([0,1,0],math.radians(30)))
	tool = tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))
	robot=abb6640(d=50)
	# data_dir="fitting_output_new/all_theta_opt/"
	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	# data = read_csv(data_dir+"/arm1.csv", names=col_names)
	# data = read_csv("../../../data/8/Curve_js.csv", names=col_names)
	data = read_csv("../../../constraint_solver/single_arm/trajectory/stepwise_opt/arm1.csv", names=col_names)

	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	step=int(len(curve_js)/500)

	v700 = speeddata(700,500,5000,1000)
	speed=v500
	mp = MotionProgram()
	j0=jointtarget(np.degrees(curve_js[0]),[0]*6)
	mp.MoveAbsJ(j0,v500,fine)
	mp.WaitTime(1)
	mp.MoveAbsJ(j0,v50,fine)
	for i in range(1,len(curve_js),step):
		j = jointtarget(np.degrees(curve_js[i]),[0]*6)
		mp.MoveAbsJ(j,speed,z10)


	jf=jointtarget(np.degrees(curve_js[-1]),[0]*6)
	mp.MoveAbsJ(jf,speed,fine)
	mp.WaitTime(1)

	print(mp.get_program_rapid())

	client = MotionProgramExecClient()
	log_results = client.execute_motion_program(mp)

	# Write log csv to file
	with open("log.csv","wb") as f:
	   f.write(log_results)

if __name__ == "__main__":
	main()



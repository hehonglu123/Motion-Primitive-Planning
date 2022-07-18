from pandas import read_csv
from abb_motion_program_exec_client import *
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *

def main():
	quatR = R2q(rot([0,1,0],math.radians(30)))
	tool = tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))
	# data_dir="fitting_output_new/all_theta_opt/"
	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	# train_data = read_csv(data_dir+"/arm1.csv", names=col_names)
	# train_data = read_csv("../../../train_data/8/Curve_js.csv", names=col_names)
	data1 = read_csv("../../../constraint_solver/dual_arm/trajectory/arm1.csv", names=col_names)
	data2 = read_csv("../../../constraint_solver/dual_arm/trajectory/arm2.csv", names=col_names)

	curve1_q1=data1['q1'].tolist()
	curve1_q2=data1['q2'].tolist()
	curve1_q3=data1['q3'].tolist()
	curve1_q4=data1['q4'].tolist()
	curve1_q5=data1['q5'].tolist()
	curve1_q6=data1['q6'].tolist()
	curve1_js=np.vstack((curve1_q1, curve1_q2, curve1_q3,curve1_q4,curve1_q5,curve1_q6)).T

	curve2_q1=data2['q1'].tolist()
	curve2_q2=data2['q2'].tolist()
	curve2_q3=data2['q3'].tolist()
	curve2_q4=data2['q4'].tolist()
	curve2_q5=data2['q5'].tolist()
	curve2_q6=data2['q6'].tolist()
	curve2_js=np.vstack((curve2_q1, curve2_q2, curve2_q3,curve2_q4,curve2_q5,curve2_q6)).T

	step=int(len(curve1_js)/50)

	v700 = speeddata(700,500,5000,1000)
	speed=v500
	mp1 = MotionProgram()
	j1_init=jointtarget(np.degrees(curve1_js[0]),[0]*6)
	mp1.MoveAbsJ(j1_init,v500,fine)
	mp1.WaitTime(1)
	mp1.MoveAbsJ(j1_init,v50,fine)

	mp2 = MotionProgram()
	j2_init=jointtarget(np.degrees(curve2_js[0]),[0]*6)
	mp2.MoveAbsJ(j2_init,v500,fine)
	mp2.WaitTime(1)
	mp2.MoveAbsJ(j2_init,v50,fine)

	for i in range(1,len(curve1_js),step):
		j1 = jointtarget(np.degrees(curve1_js[i]),[0]*6)
		mp1.MoveAbsJ(j1,speed,z10)

		j2 = jointtarget(np.degrees(curve2_js[i]),[0]*6)
		mp2.MoveAbsJ(j2,speed,z10)


	jf1=jointtarget(np.degrees(curve1_js[-1]),[0]*6)
	mp1.MoveAbsJ(jf1,speed,fine)
	mp1.WaitTime(1)

	jf2=jointtarget(np.degrees(curve2_js[-1]),[0]*6)
	mp2.MoveAbsJ(jf2,speed,fine)
	mp2.WaitTime(1)

	print(mp1.get_program_rapid())

	client = MotionProgramExecClient()
	log_results = client.execute_multimove_motion_program([mp1,mp2])

	# Write log csv to file
	with open("log.csv","wb") as f:
	   f.write(log_results)

if __name__ == "__main__":
	main()



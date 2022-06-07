from pandas import read_csv
from abb_motion_program_exec_client import *
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *

def quadrant(q):
	temp=np.ceil(np.array([q[0],q[3],q[5]])/(np.pi/2))-1
	
	if q[4] < 0:
		last = 1
	else:
		last = 0

	return np.hstack((temp,[last])).astype(int)


def main():
	quatR = R2q(rot([0,1,0],math.radians(30)))
	tool1 = tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))
	tool2 = tooldata(True,pose([50,0,450],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))
	robot1=abb1200(d=50)
	robot2=abb6640()
	base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
	base2_p=np.array([3000,1000,0])
	data_dir=""

	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
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


	step=int(len(curve1_js)/20)

	v700 = speeddata(700,500,5000,1000)

	speed={"v500":v500}
	zone={"z10":z10}

	for s in speed:
		for z in zone: 

			mp1 = MotionProgram(tool=tool1)
			j1_init=jointtarget(np.degrees(curve1_js[0]),[0]*6)
			mp1.MoveAbsJ(j1_init,v500,fine)
			mp1.WaitTime(1)
			mp1.MoveAbsJ(j1_init,v50,fine)

			mp2 = MotionProgram(tool=tool2)
			j2_init=jointtarget(np.degrees(curve2_js[0]),[0]*6)
			mp2.MoveAbsJ(j2_init,v500,fine)
			mp2.WaitTime(1)
			mp2.MoveAbsJ(j2_init,v50,fine)

			for i in range(1,len(curve1_js),step):
				pose1_now=robot1.fwd(curve1_js[i])
				pose2_now=robot2.fwd(curve2_js[i])

				pose2_world_now=robot2.fwd(curve2_js[i],base2_R,base2_p)

				quatR1=R2q(pose1_now.R)
				cf1=quadrant(curve1_js[i])
				r1 = robtarget(pose1_now.p, quatR1, confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[0]*6)
				mp1.MoveL(r1,speed[s],zone[z])

				quatR2=R2q(pose2_world_now.R)
				cf2=quadrant(curve2_js[i])
				r2 = robtarget(pose2_world_now.p, quatR2, confdata(cf2[0],cf2[1],cf2[2],cf2[3]),[0]*6)
				mp2.MoveL(r2,speed[s],zone[z])



			jf1=jointtarget(np.degrees(curve1_js[-1]),[0]*6)
			mp1.MoveAbsJ(jf1,speed[s],fine)
			mp1.WaitTime(1)

			jf2=jointtarget(np.degrees(curve2_js[-1]),[0]*6)
			mp2.MoveAbsJ(jf2,speed[s],fine)
			mp2.WaitTime(1)

			print(mp1.get_program_rapid())

			client = MotionProgramExecClient()
			log_results = client.execute_multimove_motion_program([mp1,mp2])

			# Write log csv to file
			with open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv","wb") as f:
			   f.write(log_results)

if __name__ == "__main__":
	main()
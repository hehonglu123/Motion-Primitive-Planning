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
	tool = tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))
	robot=abb6640(d=50)
	
	data_dir="../../../data/wood/"

	###read actual curve
	curve_js = read_csv(data_dir+'Curve_js.csv',header=None).values


	# data_dir="fitting_output_new/python_qp_movel/"
	# col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	# data = read_csv(data_dir+"arm1.csv", names=col_names)
	# curve_q1=data['q1'].tolist()
	# curve_q2=data['q2'].tolist()
	# curve_q3=data['q3'].tolist()
	# curve_q4=data['q4'].tolist()
	# curve_q5=data['q5'].tolist()
	# curve_q6=data['q6'].tolist()
	# curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T


	step=int(len(curve_js)/50)

	v180 = speeddata(180,500,5000,1000)

	speed={"v180":v180}
	zone={"z10":z10}

	for s in speed:
		for z in zone: 

			mp = MotionProgram(tool=tool)
			j0=jointtarget(np.degrees(curve_js[0]),[0]*6)
			mp.MoveAbsJ(j0,v500,fine)
			mp.WaitTime(1)
			mp.MoveAbsJ(j0,v50,fine)
			for i in range(1,len(curve_js),step):
				pose_temp=robot.fwd(curve_js[i])
				quatR=R2q(pose_temp.R)
				cf=quadrant(curve_js[i])
				r = robtarget(pose_temp.p, quatR, confdata(cf[0],cf[1],cf[2],cf[3]),[0]*6)
				mp.MoveL(r,speed[s],zone[z])


			jf=jointtarget(np.degrees(curve_js[-1]),[0]*6)
			mp.MoveAbsJ(jf,speed[s],fine)
			mp.WaitTime(1)

			print(mp.get_program_rapid())

			client = MotionProgramExecClient()
			log_results = client.execute_motion_program(mp)

			# Write log csv to file
			with open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv","wb") as f:
			   f.write(log_results)

if __name__ == "__main__":
	main()
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
	data_dir='tesseract/'
	curve_js = read_csv(data_dir+'movej_waypoints.csv',header=None).values[:,1:]

	step=int(len(curve_js)/500)

	v700 = speeddata(700,500,5000,1000)
	speed={'v500':v500}
	zone={'z10':z10}
	for s in speed:
		for z in zone:
			mp = MotionProgram()
			j0=jointtarget(np.degrees(curve_js[0]),[0]*6)
			mp.MoveAbsJ(j0,v500,fine)
			mp.WaitTime(1)
			mp.MoveAbsJ(j0,v50,fine)
			for i in range(1,len(curve_js),step):
				j = jointtarget(np.degrees(curve_js[i]),[0]*6)
				mp.MoveAbsJ(j,speed[s],zone[z])


			jf=jointtarget(np.degrees(curve_js[-1]),[0]*6)
			mp.MoveAbsJ(jf,speed[s],zone[z])
			mp.WaitTime(1)

			print(mp.get_program_rapid())

			client = MotionProgramExecClient()
			log_results = client.execute_motion_program(mp)

			# Write log csv to file
			with open(data_dir+"curve_exe_"+s+'_'+z+".csv","wb") as f:
			   f.write(log_results)

if __name__ == "__main__":
	main()



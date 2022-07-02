from pandas import read_csv
from abb_motion_program_exec_client import *
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *

def main():
	q1=np.array([-0.58589046, -0.03341733,  0.40482559, -2.79157683, -1.61501501,  0.95669868])
	q2=np.array([1.68077234e-02, -5.74241417e-02,  5.87495788e-02,  1.56731523e-03, -1.71647274e+00, -1.57941016e+00])
	mp1 = MotionProgram()
	j1_init=jointtarget(np.degrees(q1),[0]*6)
	mp1.MoveAbsJ(j1_init,v500,fine)
	mp1.WaitTime(1)
	mp1.MoveAbsJ(j1_init,v50,fine)

	mp2 = MotionProgram()
	j2_init=jointtarget(np.degrees(q2),[0]*6)
	mp2.MoveAbsJ(j2_init,v500,fine)
	mp2.WaitTime(1)
	mp2.MoveAbsJ(j2_init,v50,fine)

	print(mp1.get_program_rapid())

	client = MotionProgramExecClient()
	log_results = client.execute_multimove_motion_program([mp1,mp2])

	# Write log csv to file
	with open("log.csv","wb") as f:
	   f.write(log_results)

if __name__ == "__main__":
	main()



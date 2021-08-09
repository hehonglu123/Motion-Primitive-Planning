import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *

sys.path.append('../toolbox')
from robot_def import *

def cross(v):
	return np.array([[0,-v[-1],v[1]],
					[v[-1],0,-v[0]],
					[-v[1],v[0],0]])
def direction2R(v_norm,v_tang):
	v_norm=v_norm/np.linalg.norm(v_norm)
	theta1 = np.arccos(np.dot(np.array([0,0,1]),v_norm))
	###rotation to align z axis with curve normal
	axis_temp=np.cross(np.array([0,0,1]),v_norm)
	R1=rot(axis_temp/np.linalg.norm(axis_temp),theta1)

	###find correct x direction
	v_temp=v_tang-v_norm * np.dot(v_tang, v_norm) / np.linalg.norm(v_norm)

	###get as ngle to rotate
	theta2 = np.arccos(np.dot(R1[:,0],v_temp/np.linalg.norm(v_temp)))


	axis_temp=np.cross(R1[:,0],v_temp)
	axis_temp=axis_temp/np.linalg.norm(axis_temp)

	###rotation about z axis to minimize x direction error
	R2=rot(np.array([0,0,np.sign(np.dot(axis_temp,v_norm))]),theta2)


	return np.dot(R1,R2)


def main():
	###reference frame transformation
	R=np.array([[0,0,1.],
				[1.,0,0],
				[0,1.,0]])
	T=np.array([[2700.],[-800.],[500.]])
	H=np.vstack((np.hstack((R,T)),np.array([0,0,0,1])))



	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("Curve_dense.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z))

	###back projection
	d=50			###offset
	curve=curve-d*curve_direction.T



	#convert curve direction to base frame
	curve_direction=np.dot(R,curve_direction).T





	curve_base=np.zeros(curve.shape)
	curve_R_base=[]


	for i in range(len(curve)):
		curve_base[i]=np.dot(H,np.hstack((curve[i],[1])).T)[:-1]
		try:
			R_curve=direction2R(curve_direction[i],-curve_base[i]+curve_base[i-1])

		except:
			traceback.print_exc()
			pass
		curve_R_base.append(R_curve)

	###insert initial orientation
	curve_R_base.insert(0,curve_R_base[0])

	###units
	curve_base=curve_base/1000.

	curve_js=np.zeros((len(curve),6))
	curve_js1=np.zeros((len(curve),6))

	# q_init=np.radians([33.340200, 19.794526, 36.587148, -140.737677, 79.139957, -177.061128])
	q_init=np.array([0.627463700138299,0.17976842821744082,0.5196590573281621,1.6053098733278601,-0.8935105128511388,0.9174696574156079])
	for i in range(len(curve_base)):
		try:
			q_all=np.array(inv(curve_base[i],curve_R_base[i]))
		except:
			pass
		###choose inv_kin closest to previous joints
		if i==0:
			temp_q=q_all-q_init
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			curve_js[i]=q_all[order[0]]

		else:
			try:
				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				curve_js[i]=q_all[order[0]]

			except:
				pass

	###checkpoint3
	###make sure fwd(joint) and original curve match
	# H=np.vstack((np.hstack((R.T,-np.dot(R.T,T))),np.array([0,0,0,1])))
	# curve_base_temp=np.zeros(curve.shape)
	# for i in range(len(curve_js)):
	# 	curve_base_temp[i]=(np.dot(H,np.hstack((1000.*fwd(curve_js[i]).p,[1])).T)[:-1])
	# print(np.max(np.linalg.norm(curve-curve_base_temp,axis=1)))




	###output to csv
	df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
	df.to_csv('Curve_backproj_js.csv',header=False,index=False)




if __name__ == "__main__":
	main()
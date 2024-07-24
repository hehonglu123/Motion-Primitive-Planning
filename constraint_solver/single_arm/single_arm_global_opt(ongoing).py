import sys
sys.path.append('../../toolbox')
from robot_def import *
sys.path.append('../')
from constraint_solver import *


def main():
	dataset='curve_2/'
	data_dir='../../data/'+dataset+'/'
	solution_dir='baseline/'
	curve = read_csv(data_dir+solution_dir+"Curve_in_base_frame.csv",header=None).values
	curve_js = read_csv(data_dir+solution_dir+"Curve_js.csv",header=None).values

	robot=robot_obj('ABB_6640_180_255','../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/acceleration/6640acc_new.pickle')


	###formulate equality constraint
	for i in range(len(curve)):
		# print(i)
		try:
			now=time.time()
			error_fb=999
			error_fb_prev=999
			error_fb_w=999

			while error_fb>0.0001 and error_fb_w>0.0000000001:
				if time.time()-now>10:
					print('qp timeout')
					raise AssertionError
					break

				pose_now=self.robot1.fwd(q_all[-1])
				error_fb=np.linalg.norm(pose_now.p-curve[i])+Kw*np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
				error_fb_w=np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
				
				Kq=.01*np.eye(6)    #small value to make sure positive definite
				KR=np.eye(3)        #gains for position and orientation error

				vd=curve[i]-pose_now.p
				J=self.robot1.jacobian(q_all[-1])        #calculate current Jacobian
				Jp=J[3:,:]
				JR=J[:3,:]
				JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

				
				if using_spherical:
					###using spherical coordinates
					JR_mod=dndspherical_Jacobian(pose_now.R[:,-1],JR_mod)
					dspherical_des=Cartesian2Spherical(curve_normal[i])-Cartesian2Spherical(pose_now.R[:,-1])
					f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),dspherical_des)
				else:
					###using ezcross
					ezdotd=(curve_normal[i]-pose_now.R[:,-1])
					f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),ezdotd)

				H=np.dot(np.transpose(Jp),Jp)+Kq+Kw*np.dot(np.transpose(JR_mod),JR_mod)
				H=(H+np.transpose(H))/2
				qdot=solve_qp(H,f,lb=(self.robot1.lower_limit+0.1)-q_all[-1]+self.lim_factor*np.ones(6),ub=(self.robot1.upper_limit-0.1)-q_all[-1]-self.lim_factor*np.ones(6),solver='cvxopt')
				
				#avoid getting stuck
				if abs(error_fb-error_fb_prev)<0.0001:
					raise AssertionError("Stuck")

				error_fb_prev=error_fb

				###line search
				alpha=fminbound(self.error_calc,0,0.999999999999999999999,args=(q_all[-1],qdot,curve[i],curve_normal[i],))
				if alpha<0.01:
					break
				q_all.append(q_all[-1]+alpha*qdot)
				# print(q_all[-1])
		except:
			q_out.append(q_all[-1])
			traceback.print_exc()
			raise AssertionError
			break

		q_out.append(q_all[-1])

	q_out=np.array(q_out)[1:]
	return q_out


if __name__ == '__main__':
	main()
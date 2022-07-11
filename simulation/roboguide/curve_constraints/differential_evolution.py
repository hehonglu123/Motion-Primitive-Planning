import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../../../constraint_solver')
from constraint_solver import *


def main():
    dataset='wood'
    # dataset='blade'
    curve = read_csv("../ilc_fanuc/data/"+dataset+"/Curve_in_base_frame.csv",header=None).values


    opt=lambda_opt(curve[:,:3],curve[:,3:],robot1=m710ic(d=50),steps=50)

    ###path constraints, position constraint and curve normal constraint
    lowerer_limit=[-np.pi]
    upper_limit=[np.pi]
    bnds=tuple(zip(lowerer_limit,upper_limit))*len(opt.curve)
    bnds=tuple(zip([0],[8]))+bnds


    ###diff evolution
    res = differential_evolution(opt.single_arm_global_opt, bnds,workers=-1,
                                    x0 = np.zeros(len(opt.curve)+1),
                                    strategy='best1bin', maxiter=2000,
                                    popsize=15, tol=1e-2,
                                    mutation=(0.5, 1), recombination=0.7,
                                    seed=None, callback=None, disp=False,
                                    polish=True, init='latinhypercube',
                                    atol=0.)

    print(res)
    theta=res.x[1:]
    pose_choice=int(np.floor(res.x[0]))

    for i in range(len(opt.curve)):
        if i==0:
            R_temp=direction2R(opt.curve_normal[0],-opt.curve_original[1]+opt.curve[0])
            R=np.dot(R_temp,Rz(theta[i]))
            try:
                q_out=[opt.robot1.inv(opt.curve[i],R)[pose_choice]]
            except:
                traceback.print_exc()
                return 999
        else:
            R_temp=direction2R(opt.curve_normal[i],-opt.curve[i]+opt.curve_original[opt.act_breakpoints[i]-1])

            R=np.dot(R_temp,Rz(theta[i]))
            try:
                ###get closet config to previous one
                q_inv_all=opt.robot1.inv(opt.curve[i],R)
                temp_q=q_inv_all-q_out[-1]
                order=np.argsort(np.linalg.norm(temp_q,axis=1))
                q_out.append(q_inv_all[order[0]])
            except:
                # traceback.print_exc()
                return 999

    q_out=np.array(q_out)
    ####output to trajectory csv
    df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    df.to_csv('data/'+dataset+'_de_arm1.csv',header=False,index=False)


    dlam_out=calc_lamdot(q_out,opt.lam[:len(q_out)],opt.robot1,1)

    ###############################################restore 50,000 points#############################################
    # theta_all=[]
    # for i in range(len(res.x)-1):
    # 	theta_all=np.append(theta_all,np.linspace(res.x[i],res.x[i+1],(opt.idx[i+1]-opt.idx[i])))
    # theta_all=np.append(theta_all,res.x[-1]*np.ones(len(curve)-len(theta_all)))

    # for i in range(len(theta_all)):
    # 	if i==0:
    # 		R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
    # 		R=np.dot(R_temp,Rz(theta_all[i]))
    # 		q_out=[opt.robot1.inv(curve[i],R)[0]]

    # 	else:
    # 		R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
    # 		R=np.dot(R_temp,Rz(theta_all[i]))
    # 		###get closet config to previous one
    # 		q_inv_all=opt.robot1.inv(curve[i],R)
    # 		temp_q=q_inv_all-q_out[-1]
    # 		order=np.argsort(np.linalg.norm(temp_q,axis=1))
    # 		q_out.append(q_inv_all[order[0]])

    # q_out=np.array(q_out)

    # ###output to csv
    # df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    # df.to_csv('trajectory/all_theta_opt/all_theta_opt_js.csv',header=False,index=False)
    ####################################################################################################################

    plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
    plt.xlabel("lambda")
    plt.ylabel("lambda_dot")
    plt.ylim([0,2000])
    plt.title("max lambda_dot vs lambda (path index)")
    plt.savefig("data/"+dataset+"_de_arm1.png")
    plt.show()

    


if __name__ == "__main__":
    main()
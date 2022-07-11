import sys
from pandas import *
from matplotlib import pyplot as plt
sys.path.append('../../../constraint_solver')
from constraint_solver import *


def main():
    # dataset='wood'
    dataset='blade'

    curve = read_csv("../ilc_fanuc/data/"+dataset+"/Curve_in_base_frame.csv",header=None).values
    curve_js = read_csv("../ilc_fanuc/data/"+dataset+"/Curve_js.csv",header=None).values

    robot=m710ic(d=50)
    opt=lambda_opt(curve[:,:3],curve[:,3:],robot1=robot,steps=10000)
    q_init=curve_js[0]
    # q_init=[-1.71E-06,	0.458127951,	0.800479092,	-5.65E-06,	-1.782205819,	-7.34E-06]

    q_out=opt.single_arm_stepwise_optimize(q_init)

    ####output to trajectory csv
    df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    df.to_csv('data/'+dataset+'_qp_arm1.csv',header=False,index=False)

    dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)


    plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
    plt.xlabel("lambda")
    plt.ylabel("lambda_dot")
    plt.ylim([0,2000])
    plt.title("max lambda_dot vs lambda (path index)")
    plt.savefig("data/"+dataset+"_qp_arm1.png")
    plt.show()
    

if __name__ == "__main__":
    main()
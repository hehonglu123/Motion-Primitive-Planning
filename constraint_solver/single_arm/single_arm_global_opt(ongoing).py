import sys, argparse
import numpy as np
from pandas import read_csv
from scipy.optimize import basinhopping, differential_evolution, shgo, dual_annealing, direct, Bounds, NonlinearConstraint

sys.path.append('../../toolbox')
from robot_def import robot_obj
sys.path.append('../')
from constraint_solver import lambda_opt

def main():
	dataset = 'curve_1/'
	data_dir = '../../data/' + dataset + '/'
	solution_dir = 'baseline/'
	curve = read_csv(data_dir + solution_dir + "Curve_in_base_frame.csv", header=None).values

	robot = robot_obj('ABB_6640_180_255', '../../config/ABB_6640_180_255_robot_default_config.yml', tool_file_path='../../config/paintgun.csv', d=50, acc_dict_path='../../config/acceleration/6640acc_new.pickle')

	steps = 100
	opt = lambda_opt(curve[:, :3], curve[:, 3:], robot1=robot, curve_name=dataset[:-1], steps=steps)
	q_seed = np.loadtxt(data_dir + solution_dir + 'Curve_js.csv', delimiter=',')
	q_seed = q_seed[opt.act_breakpoints]


	eq_constraint = NonlinearConstraint(lambda q: opt.single_arm_constraint(q), 0.0, 0.1)
	lim_factor=1e-4
	bounds = Bounds((robot.lower_limit+lim_factor).tolist() * len(opt.curve), (robot.upper_limit-lim_factor).tolist() * len(opt.curve))


	# print("EQ CONSTRAINT TEST: ",opt.single_arm_constraint(q_seed.flatten()))
	# print("OBJECTIVE TEST: ",opt.single_arm_objective(q_seed.flatten()))

	# Set up argument parser
	parser = argparse.ArgumentParser(description='Run optimization methods.')
	parser.add_argument('--method', type=str, required=True, choices=['basinhopping', 'differential_evolution', 'shgo', 'dual_annealing', 'direct'],
						help='Optimization method to use.')

	args = parser.parse_args()

	if args.method == 'basinhopping':
		local_minimizer = {
			'method': 'SLSQP',
			'bounds': bounds,
			'constraints': eq_constraint
		}
		result = basinhopping(opt.single_arm_objective, q_seed.flatten(), minimizer_kwargs=local_minimizer,
							niter=3000, T=1.0, stepsize=0.5, interval=50, disp=True)

	elif args.method == 'differential_evolution':
		result = differential_evolution(opt.single_arm_objective, bounds, constraints=(eq_constraint), args=None, workers=10,
										x0=q_seed.flatten(),
										strategy='best1bin', maxiter=3000,
										popsize=15, tol=1e-10,
										mutation=(0.5, 1), recombination=0.7,
										seed=None, callback=None, disp=True,
										polish=True, init='latinhypercube',
										atol=0)

	elif args.method == 'shgo':
		eq_constraint_dict = {
			'type': 'eq',
			'fun': lambda q: opt.single_arm_constraint(q),
			'lb': 0.0,
			'ub': 0.1
		}
		result = shgo(opt.single_arm_objective, bounds, constraints=[eq_constraint_dict],
					n=100, iters=5, sampling_method='sobol', options={'disp': True})

	elif args.method == 'dual_annealing':
		result = dual_annealing(opt.single_arm_objective, bounds,
								maxiter=3000, initial_temp=5230.0, restart_temp_ratio=2e-5,
								visit=2.62, accept=-5.0, maxfun=1e7, seed=None,
								no_local_search=False, callback=None, x0=q_seed.flatten())

	#direct method not accepts constraint only bounds

if __name__ == '__main__':
	main()
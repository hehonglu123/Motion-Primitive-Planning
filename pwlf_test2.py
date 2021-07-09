from GPyOpt.methods import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import pwlf

x=np.arange(-2,2,0.001)
y=np.sin(x)


# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# define your objective function


def my_obj(x):
    # define some penalty parameter l
    # you'll have to arbitrarily pick this
    # it depends upon the noise in your data,
    # and the value of your sum of square of residuals
    l = y.mean()*0.001
    f = np.zeros(x.shape[0])
    for i, j in enumerate(x):
        my_pwlf.fit(j[0])
        f[i] = my_pwlf.ssr + (l*j[0])
    return f


# define the lower and upper bound for the number of line segments
bounds = [{'name': 'var_1', 'type': 'discrete',
           'domain': np.arange(2, 40)}]

np.random.seed(12121)

myBopt = BayesianOptimization(my_obj, domain=bounds, model_type='GP',
                              initial_design_numdata=10,
                              initial_design_type='latin',
                              exact_feval=True, verbosity=True,
                              verbosity_model=False)
max_iter = 30

# perform the bayesian optimization to find the optimum number
# of line segments
myBopt.run_optimization(max_iter=max_iter, verbosity=True)

print('\n \n Opt found \n')
print('Optimum number of line segments:', myBopt.x_opt)
print('Function value:', myBopt.fx_opt)
myBopt.plot_acquisition()
myBopt.plot_convergence()

# perform the fit for the optimum
my_pwlf.fit(myBopt.x_opt)
# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(xHat)

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.show()
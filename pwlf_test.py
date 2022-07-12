
import numpy as np
import matplotlib.pyplot as plt
import pwlf

x=np.arange(-2,2,0.001)
y=np.sin(x)

# initialize piecewise linear fit with your x and y train_data
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# fit the train_data for four line segments
res = my_pwlf.fit(4)

# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(xHat)

# plot the results
plt.figure()
plt.title('fitting plot')
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')

plt.figure(2)
plt.title('error plot')

plt.plot(xHat,np.abs(yHat-np.sin(xHat)))
plt.show()
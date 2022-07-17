import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": '3d'})
for i in range(101, 102):
    curve = pd.read_csv('eval_data/curve1/forward/base/curve_{}.csv'.format(i), header=None).values
    ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2])
plt.show()


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = [(2.2176383052987667, 4.218574252410221),
(3.3041214516913033, 5.223500807396272),
(4.280815855023374, 6.461487709813785),
(4.946375258539319, 7.606952538212697),
(5.382428804463699, 9.045717060494576),
(5.752578028217334, 10.613667377465823),
(5.547729017414035, 11.92662513852466),
(5.260208374620305, 13.57722448066025),
(4.642126672822957, 14.88238955729078),
(3.820310290976751, 16.10605425390148),
(2.8099420132544024, 17.225880123445773),
(1.5731539516426183, 18.17052077121059),
(0.31752822350872545, 18.75261434891438),
(-1.2408437559671106, 19.119355580780265),
(-2.680901948575409, 19.15018791257732),
(-4.190406775175328, 19.001321726517297),
(-5.533990404926917, 18.64857428377178),
(-6.903383826792998, 17.730112542165955),
(-8.082883753215347, 16.928080323602334),
(-9.138397388219254, 15.84088004983959),
(-9.92610373064812, 14.380575762984085),
(-10.358670204629814, 13.018017342781242),
(-10.600053524240247, 11.387283417089911),
(-10.463673966507077, 10.107554951600699),
(-10.179820255235496, 8.429558128401448),
(-9.572153386953028, 7.1976672709797676),
(-8.641475289758178, 5.8312286526738175),
(-7.665976739804268, 4.782663065707469),
(-6.493033077746997, 3.8549965442534684),
(-5.092340806635571, 3.384419909199452),
(-3.6530364510489073, 2.992272643733981),
(-2.1522365767310796, 3.020780664301393),
(-0.6855406924835704, 3.0767643753777447),
(0.7848958776292426, 3.6196842530995332),
(2.0614188482646947, 4.32795711960546),
(3.2705467984691508, 5.295836809444288),
(4.359297538484424, 6.378324784240816),
(4.981264502955681, 7.823851404553242)]

data=np.array(data).T
fun = lambda x: np.linalg.norm(x[0]*data[0] + x[1]*data[1] + data[0][0]**2 - x[0]*data[0][0] + data[1][0]**2 - x[1]*data[1][0] - np.square(data[0]) - np.square(data[1]))
res = minimize(fun, (0,0), method='SLSQP')

center=res.x/2.
r=np.sqrt((data[0][0]-center[0])**2+(data[1][0]-center[1])**2)

circle1 = plt.Circle(center, r, color='r',fill=False)


fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

ax.add_patch(circle1)
ax.set_xlim((-20, 20))
ax.set_ylim((-20, 20))
plt.plot(data[0],data[1])
plt.plot(center[0],center[1],'-go')

plt.show()

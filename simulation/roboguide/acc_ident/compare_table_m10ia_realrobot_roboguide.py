import pickle
import matplotlib.pyplot as plt
import numpy as np

def get_nearest_acc(q_all,q2q3_config,q1q2q3_acc):
   ###find closest q2q3 config, along with constant last 3 joints acc
   idx=np.argmin(np.linalg.norm(q2q3_config-q_all,axis=1))
   return q1q2q3_acc[idx]

# dic = ''
# with open(r'test.txt','r') as f:
#          for i in f.readlines():
#             dic=i #string
# dic = eval(dic) # this is orignal dict with instace dict

dic_roboguide = pickle.load(open('m10ia/m10ia_acc_shake.pickle','rb'))
dic_realrobot = pickle.load(open('m10ia_realrobot/m10ia_acc_shake.pickle','rb'))

###surface plots of accleration limits, x as q2, y as q3
###surface plots of accleration limits, x as q2, y as q3
q2q3_roboguide=np.array(list(dic_roboguide.keys()))
q2q3_roboguide_acc=[]
for key, value in dic_roboguide.items():
   closest_id=np.argsort(np.linalg.norm(q2q3_roboguide-np.array(list(key)),axis=1))
   acc=np.zeros(6)

   for acc_i in range(6):
      closest_id_i=0
      while True:
         if dic_roboguide[tuple(q2q3_roboguide[closest_id[closest_id_i]])][acc_i]!=0:
            acc[acc_i]=dic_roboguide[tuple(q2q3_roboguide[closest_id[closest_id_i]])][acc_i]
            break
         closest_id_i+=1
   q2q3_roboguide_acc.append(acc)
q2q3_roboguide_acc=np.array(q2q3_roboguide_acc)

q2q3_realrobot=np.array(list(dic_realrobot.keys()))
q2q3_realrobot_acc=[]

q2q3_diff_acc=[]

for key, value in dic_realrobot.items():
   closest_id=np.argsort(np.linalg.norm(q2q3_realrobot-np.array(list(key)),axis=1))
   acc=np.zeros(6)

   for acc_i in range(6):
      closest_id_i=0
      while True:
         if dic_realrobot[tuple(q2q3_realrobot[closest_id[closest_id_i]])][acc_i]!=0:
            acc[acc_i]=dic_realrobot[tuple(q2q3_realrobot[closest_id[closest_id_i]])][acc_i]
            break
         closest_id_i+=1
   q2q3_realrobot_acc.append(acc)
   
   roboguide_closest_id=np.argsort(np.linalg.norm(q2q3_roboguide-np.array(list(key)),axis=1))
   acc_diff = acc-q2q3_roboguide_acc[roboguide_closest_id[0]]
   acc_diff = np.divide(acc_diff,acc)*100
   q2q3_diff_acc.append(acc_diff)
q2q3_realrobot_acc=np.array(q2q3_realrobot_acc)
q2q3_diff_acc=np.array(q2q3_diff_acc)

#####################################################################get acc from q###########################################################
# q=np.array([2,0,-1,1,3,4])
# xy=np.array([x,y]).T
# idx=np.argmin(np.linalg.norm(xy-q[1:3],axis=1))
# print('q2,q3 at: ',x[idx],y[idx])
# print('acc: ',q1_acc[idx],q2_acc[idx],q3_acc[idx],47.29253791291949,39.49167516506145,54.32806813314554)

#####################################################################surface plots##########################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_roboguide = ax.plot_trisurf(q2q3_roboguide[:,0], q2q3_roboguide[:,1], q2q3_roboguide_acc[:,0],alpha=0.5, linewidth=0, antialiased=False)
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_realrobot_acc[:,0], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q1 acc (rad/s^2)')

plt.title('Joint1 Pos-Direction Acceleration Limit')
plt.show()
# exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_roboguide = ax.plot_trisurf(q2q3_roboguide[:,0], q2q3_roboguide[:,1], q2q3_roboguide_acc[:,1],alpha=0.5, linewidth=0, antialiased=False)
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_realrobot_acc[:,1], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q1 acc (rad/s^2)')

plt.title('Joint1 Neg-Direction Acceleration Limit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_diff_acc[:,0], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q1 acc diff (%)')
plt.title('Joint1 Pos Acceleration Limit Differences (Roboguide v.s. Realrobot)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_diff_acc[:,1], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q1 acc diff (%)')
plt.title('Joint1 Neg Acceleration Limit Differences (Roboguide v.s. Realrobot)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_roboguide = ax.plot_trisurf(q2q3_roboguide[:,0], q2q3_roboguide[:,1], q2q3_roboguide_acc[:,2],alpha=0.5, linewidth=0, antialiased=False)
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_realrobot_acc[:,2], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q2 acc (rad/s^2)')

plt.title('Joint2 Pos-Direction Acceleration Limit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_roboguide = ax.plot_trisurf(q2q3_roboguide[:,0], q2q3_roboguide[:,1], q2q3_roboguide_acc[:,3],alpha=0.5, linewidth=0, antialiased=False)
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_realrobot_acc[:,3], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q2 acc (rad/s^2)')

plt.title('Joint2 Neg-Direction Acceleration Limit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_diff_acc[:,2], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q2 acc diff (%)')
plt.title('Joint2 Pos Acceleration Limit Differences (Roboguide v.s. Realrobot)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_diff_acc[:,3], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q2 acc diff (%)')
plt.title('Joint2 Neg Acceleration Limit Differences (Roboguide v.s. Realrobot)')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_roboguide = ax.plot_trisurf(q2q3_roboguide[:,0], q2q3_roboguide[:,1], q2q3_roboguide_acc[:,4],alpha=0.5, linewidth=0, antialiased=False)
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_realrobot_acc[:,4], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q3 acc (rad/s^2)')

plt.title('Joint3 Pos-Direction Acceleration Limit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_roboguide = ax.plot_trisurf(q2q3_roboguide[:,0], q2q3_roboguide[:,1], q2q3_roboguide_acc[:,5],alpha=0.5, linewidth=0, antialiased=False)
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_realrobot_acc[:,5], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q3 acc (rad/s^2)')

plt.title('Joint3 Neg-Direction Acceleration Limit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_diff_acc[:,4], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q3 acc diff (%)')
plt.title('Joint3 Pos Acceleration Limit Differences (Roboguide v.s. Realrobot)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf_realrobot = ax.plot_trisurf(q2q3_realrobot[:,0], q2q3_realrobot[:,1], q2q3_diff_acc[:,5], linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q3 acc diff (%)')
plt.title('Joint3 Neg Acceleration Limit Differences (Roboguide v.s. Realrobot)')
plt.show()


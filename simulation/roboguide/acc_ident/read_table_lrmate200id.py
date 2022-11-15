import pickle
import matplotlib.pyplot as plt
import numpy as np
import ast

# dic = ''
# with open(r'test.txt','r') as f:
#          for i in f.readlines():
#             dic=i #string
# dic = eval(dic) # this is orignal dict with instace dict

# dic={}
# try:
#    dic = pickle.load(open('lrmate200id/acc.pickle','rb'))
# except EOFError:
#    print("EOF Error")
#    with open('lrmate200id/acc.txt', 'r+') as handle:
#       data = handle.read()
#    dic = ast.literal_eval(data)
dic = pickle.load(open('lrmate200id/lrmate200id_acc.pickle','rb'))

for i in range(1,2):
   this_dic=pickle.load(open('lrmate200id/acc'+str(i)+'.pickle','rb'))
   dic.update(this_dic)

# pickle.dump(dic, open('lrmate200id/lrmate200id_acc.pickle','wb'))

dic_comp={}
for key, value in dic.items():
   dic_comp[(key[0],-key[1])]=value

pickle.dump(dic_comp, open('lrmate200id/lrmate200id_acc_compensate.pickle','wb'))

###surface plots of accleration limits, x as q2, y as q3
x=[]
y=[]
q1_acc=[]
q2_acc=[]
q3_acc=[]
for key, value in dic.items():
   x.append(key[0])
   y.append(key[1])
   q1_acc.append(value[0])
   q2_acc.append(value[1])
   q3_acc.append(value[2])

#####################################################################get acc from q###########################################################
# q=np.array([2,0,-1,1,3,4])
# xy=np.array([x,y]).T
# idx=np.argmin(np.linalg.norm(xy-q[1:3],axis=1))
# print('q2,q3 at: ',x[idx],y[idx])
# print('acc: ',q1_acc[idx],q2_acc[idx],q3_acc[idx],47.29253791291949,39.49167516506145,54.32806813314554)

#####################################################################surface plots##########################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(x, y, q1_acc, linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q1 acc (rad/s^2)')

plt.title('Joint1 Acceleration Limit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(x, y, q2_acc, linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q2 acc (rad/s^2)')

plt.title('Joint2 Acceleration Limit')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(x, y, q3_acc, linewidth=0, antialiased=False)
ax.set_xlabel('q2 (rad)')
ax.set_ylabel('q3 (rad)')
ax.set_zlabel('q3 acc (rad/s^2)')

plt.title('Joint3 Acceleration Limit')
plt.show()
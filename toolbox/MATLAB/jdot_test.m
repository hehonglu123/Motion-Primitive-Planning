
ex=[1;0;0];
ey=[0;1;0];
ez=[0;0;1];
p0=[0;0;0.63];
p1=[0.6;0;0];
p2=[0;0;1.28];
p3=[0;0;0.2];
p4=[1.592;0;0];
p5=[0.2;0;0];
p6=[0;0;0];

robot.kin.joint_type=zeros(6);
robot.kin.H=[ez,ey,ey,ex,ey,ex];
robot.kin.P=[p0,p1,p2,p3,p4,p5,p6];
    
q=zeros(6,1);
qdot=zeros(6,1);
qddot=zeros(6,1);
nu0=zeros(6,1);
alpha0=zeros(6,1);
[Hmat,Jmat,nu,alpha,Jdotmat]=fwdkiniter_Jdot(robot,q,qdot,qddot,nu0,alpha0);
Jmat{7}
Jdotmat{7}
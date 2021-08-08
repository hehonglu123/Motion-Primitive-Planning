% example script of using collision checking and plotting in MATLAB
%
clear all;close all;

% define the ABB IRB 6640 robot

%abb120def;n=size(abb120.H,2);robdef=abb120;radius=.05;

% create collision body for the specified robot
%[rob,colLink]=collisionBody(robdef,radius);

% generate target robot pose 
% T=readtable('Curve.csv');
T=readtable('Curve_in_base_frame.csv');
%load curve

x=T.Var1;y=T.Var2;z=T.Var3;
N=length(x);
figure(1);plot3(T.Var1,T.Var2,T.Var3,'-x');hold on 
u=T.Var4;v=T.Var5;w=T.Var6;
m=500;nn=(1:m:N);
quiver3(x(nn),y(nn),z(nn),u(nn),v(nn),w(nn));grid;

p0=[x y z]'; % surface location
k0=[u v w]';
%k=[u v w]'; % unit vector
d=50; % off set 
p=[x y z]-d*[u v w];

dd=vecnorm(diff([x y z])');
disp('path differnce between consecutive surface points (max/min)');
disp([max(dd) min(dd)]);

Ze=[u v w]';
Xe=cross(Ze(:,1:end-1),diff(p)');
Xe=Xe./vecnorm(Xe);
Xe=[Xe Xe(:,end)];
Ye=cross(Ze,Xe);

for i=1:m:length(x)
    R=[Xe(:,i) Ye(:,i) Ze(:,i)];
    plotTransforms(p(i,:),R2q(R)','FrameSize',50);
end
    
% inverse kinematics

ex=[1;0;0];ey=[0;1;0];ez=[0;0;1];zv=[0;0;0];
h1=ez;h2=ey;h3=ey;h4=ex;h5=ey;h6=ex;
p01=[0,0,780]';p12=[0,0,0]';p23=[320,0,1075]';
p34=[1392.5,0,200]';p45=[0,0,0]';p56=[0,0,0]';
p6T=[200,0,0]';
abb6640.H=[h1 h2 h3 h4 h5 h6];
abb6640.P=[p01 p12 p23 p34 p45 p56 p6T];
abb6640.joint_type=[0 0 0 0 0 0]; 

q=zeros(6,N);
p1=zeros(3,N);
l=1;
for i=1:N
    qq=invelbow(abb6640,[Xe(:,i) Ye(:,i) Ze(:,i) p(i,:)'; 0 0 0 1]);
    if i==1
        q(:,i)=qq(:,l);
    else
        [dq,ind]=min(vecnorm(qq-q(:,i-1)));
        %[dq,ind]=min(vecnorm(sin((qq-q(:,i-1))/4)));
        q(:,i)=qq(:,ind);
    end
    T=fwdkinrec(1,eye(4,4),q(:,i),abb6640);
    p1(:,i)=T(1:3,4)+T(1:3,3)*d;
end
err=[x y z]'-p1;
disp('surface points error with inverse and forward kinematics (max/min)');
disp([max(vecnorm(err)) min(vecnorm(err))]);

% task space fit
diffpnorm=vecnorm(diff(diff(p))');
figure(2);plot(diffpnorm);
% joint space fit
diffqnorm=vecnorm(diff(sin(diff(q'))'/4));
figure(3);plot(diffqnorm);

K1=1;K2=400;
%K1=6000;K2=7000;
% Dq=sin((q(:,K1:K2)-q(:,K1))/4);
% [U,S,V]=svd(Dq);
% qappr=q(:,K1)+(q(:,K1:K2)-q(:,K1))*pinv(V(:,1)')*V(:,1)';
Dq=(q(:,K1:K2)-q(:,K1));
[U,S,V]=svd(Dq);
qappr=q(:,K1)+(q(:,K1:K2)-q(:,K1))*pinv(V(:,1)')*V(:,1)';
figure(5);plot(vecnorm(qappr-q(:,K1:K2)),'x')

dp=zeros(3,K2-K1+1);
for i=1:K2-K1+1
    T=fwdkinrec(1,eye(4,4),qappr(:,i),abb6640);
    pest(:,i)=T(1:3,4)+T(1:3,3)*d;
    dp(:,i)=p0(:,K1+i-1)-pest(:,i);
end
figure(6);plot(vecnorm(dp),'x');
%l=randi([1 length(x)]);T=[Xe(:,l) Ye(:,l) Ze(:,l) p(l,:)'; 0 0 0 1];
%q=invelbow(abb6640,T);T1=fwdkinrec(1,eye(4,4),q(:,1),abb6640);
%norm(T-T1)

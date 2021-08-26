%
% fwdkiniter_Jdot.m
%
% [Hmat,Jmat,nu,alpha,Jdotmat]=fwdkiniter_Jdot(robot,q,qdot,qddot,nu0,alpha0)
%
% purpose: general forward kinematics for serial chain
% 
% input:
% robot: structure contain robot kinematic parameters
%   robot.kin.H: [ h1 h2 ... hn ] axis of rotation or translation
%   robot.kin.P: [p01 p12 p23 .. p_{n-1}n p_nT] inter-link vectors
%   robot.kin.joint_type= [j1 j2 .. jn] 0 = rotational, nonzero = prismatic
% q: n-vector of rotational angle / translational displacement
% qdot: n-vector of joint velocity
% qddot: n=vector of joint acceleration
% nu0: base spatial velocity
% alpha0: base spatial acceleration
% 
% 
% output:
% Hmat{i}=[R_{0i} p_{0i}; 0 1];
% Jmat{i}= Jacobian J_i in E_i = [phi_i1 H1| phi_i2 H2 | ... | Hi]
% nu{i} = spatial velocity [omega_i;v_i] in E_i
% alpha{i} = spatial acceleration [\dot\omega_i;\dot v_i] in E_i
% Jdotmat{i} = dJ_i/dt 
%

function [Hmat,Jmat,nu,alpha,Jdotmat]=fwdkiniter_Jdot(robot,q,qdot,qddot,nu0,alpha0)

n=length(robot.kin.joint_type);
Hmat=cell(1,n+1);
Jmat=cell(1,n+1);
nu=cell(1,n+1);
alpha=cell(1,n+1);
Jdotmat=cell(1,n+1);

zv=[0;0;0];
H=[eye(3,3) zv;zv' 1];
v=nu0;
alf=alpha0;
J=[];
Jdot=[];
for i=1:n+1
   % 
   % constant arm parameters
   %
   if i<n+1;
       hi=robot.kin.H(:,i);qi=q(i);qdi=qdot(i);qddi=qddot(i);
       ji=robot.kin.joint_type(i);
   else
       qi=0;qdi=0;qddi=0;ji=0;
   end;
   Pi=robot.kin.P(:,i);   
   % 
   % Homogeneous matrix propagation
   %
   Hi=HomogTrans(qi,hi,Pi,ji);
   Hn=H*Hi;
   Hmat{i}=Hn;
   H=Hn;   
   %
   % v and alpha propagation
   %
   PHI=phi(Hi(1:3,1:3)',Hi(1:3,4));
   Hveci=Hvec(hi,ji);
   vn=PHI*v+Hveci*qdi;
   aveci=avec(v(1:3),vn(1:3),Hi(1:3,1:3)',Hi(1:3,4),hi,qdi,ji);
   alfn=PHI*alf+Hveci*qddi+aveci;
   nu{i}=vn;
   alpha{i}=alfn;
   alf=alfn;
   v=vn;
   %
   % Partial Jacobian progagation
   %
   if(~isempty(J));
       Jn=[PHI*J Hveci];
       Jdotn=-qdi*[hat(hi) zeros(3,3);zeros(3,3) hat(hi)]*Jn + ...
           PHI*[Jdot zeros(size(Hveci))];
   else;
       Jn=Hveci;
       Jdotn=zeros(size(Jn));
   end
   Jmat{i}=Jn; 
   Jdotmat{i}=Jdotn;
   J=Jn;
   Jdot=Jdotn;
end

Jmat{n+1}=Jmat{n+1}(:,1:n);
Jdotmat{n+1}=Jdotmat{n+1}(:,1:n);

%Hi=HomogTrans(0,hi,robot.kin.P(:,n+1),0);
%Hmat{n+1}=H*Hi;
%PHI=phi(eye(3,3),robot.kin.P(:,n+1));
%Jmat{n+1}=PHI*J;
end

% Coriolis and Centrifugal accelerations
function a=avec(w1,w,R,p,h,qdot,jtype)

if jtype>0
    a=[zeros(3,1);R*hat(w1)*hat(w1)*p+ 2*hat(R*w1)*h*qdot];
else
    a=[hat(R*w1)*w;R*hat(w1)*hat(w1)*p];
end

end

% spatial joint axis vector
function H=Hvec(h,jtype)

if jtype>0
    H=[zeros(3,1);h];
else
    H=[h;zeros(3,1)];
end

end

% Homogeneious matrix
function H=HomogTrans(q,h,p,jt)

if jt==0
     H=[rot(h,q) p;0 0 0 1];
else
     H=[eye(3,3) p +q * h;0 0 0 1];
end

end

% Spatial velocity progation
function Phi=phi(R,p)

Phi=[R zeros(3,3);-R*hat(p) R];

end

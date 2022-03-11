% initialization
clear all; close all;

% robot definition
ex = [1;0;0]; ey=[0;1;0]; ez=[0;0;1];
robot.P = [0.78*ez 0.32*ex 1.075*ez 0.2*ez 1.142*ex 0.2*ex 0*ex]*1000.; % in mm
robot.H = [ez ey ey ex ey ex];
robot.joint_type = [0 0 0 0 0 0];
robot.tool = [[rot(ey,deg2rad(120)) [0.45;0;-0.05]*1000.];0 0 0 1];

% import stl files
% file_name = 'test_stl_compare.stl';
file_name = 'test_stl_dense.stl';

gm = importGeometry(file_name);

data = stlread(file_name);
figure(100); trimesh(data,'FaceColor','none','EdgeColor','k'); hold; 
plotTransforms([0 0 0],[1 0 0 0],'Framesize',10); axis equal;
title("The mold (mesh)");

%data2 = stlread('generic_fan_blade.stl');
%trimesh(data2,'FaceColor','none','EdgeColor','k')

% read joint trajectory
% joint_data = readmatrix('log_cold_spray.csv');
% joint_data = readmatrix('log_cold_spray_circle.csv');
joint_data = readmatrix('log_cold_spray_lines.csv');
j_stamp = joint_data(:,1);
j_cmdnum = joint_data(:,2);
joint_p = joint_data(:,3:8)';
start_cmdnum = 3;

x_offset = 1.5;
y_offset = -1;
z_offset = 1;
offset = [x_offset;y_offset;z_offset]*1000.;

% cold spray simulation, original resolution from stl (low)
% parameters
a = 18;
sigma = 1.4;
stand_off = 25;
v0 = 400;
% v0 = 4;
line_space = 0.5;  

check_ang_margin = 88;
    
% Start cold spray deposition

p_init = data.Points';
p_progress = p_init;
p_last_layer = p_progress;
layer_end = true; % if layer end, update the point position and thus the gradient

T = [];
all_qs = [];
all_qs_nz = [];
all_v = [];

for n=1:length(joint_p(1,:))-1
    if j_cmdnum(n) < start_cmdnum
        continue
    end
    
    tic;
    
    [this_R,this_p] = fwd(robot,deg2rad(joint_p(:,n)));
    this_p = this_p - offset; % transfer to mold frame
    [next_R,next_p] = fwd(robot,deg2rad(joint_p(:,n+1)));
    next_p = next_p - offset; % transfer to mold frame
    
    all_qs = [all_qs this_p];
    
    % nz = Rs(:,n);
    % nz is pointing outward
    nz = -this_R(:,3);
    all_qs_nz = [all_qs_nz this_p+nz];
    % nx = (qs(:,n+1)-qs(:,n)-((qs(:,n+1)-qs(:,n))'*nz)*nz);
    % nx is the linear motion direction
%     (qs(:,n+1)-qs(:,n)-((qs(:,n+1)-qs(:,n))'*nz)*nz);
%     nx = (next_p-this_p);
    nx = (next_p-this_p-((next_p-this_p)'*nz)*nz);
    nx = nx/norm(nx);
    % ny = nz x nx
    ny = cross(nz,nx);
    ny = ny/norm(ny);
    ln = dot(nx,(next_p-this_p));
    delta_t = j_stamp(n+1)-j_stamp(n);
%     delta_t = delta_t*10;
    this_v = norm(next_p-this_p)/delta_t;
    all_v = [all_v this_v];
    
    if layer_end
        p_last_layer = p_progress;
        nl = point_norm(p_last_layer,data.ConnectivityList);
        layer_end = false;
    end
    
    im_ang = impact_angle(nz,nl,check_ang_margin);
    
    for p_i = 1:length(im_ang)
        if im_ang(p_i) == 0
            continue
        else
            pn_noz = [nx';ny';nz';]*(p_last_layer(:,p_i)-this_p);
            g1 = a*nz/(2*sqrt(2*pi)*sigma*ln);
            g2 = exp(-(pn_noz(2)^2)/(2*sigma^2));
            g3 = erf(pn_noz(1)/(sqrt(2)*sigma))-erf((pn_noz(1)-ln)/(sqrt(2)*sigma));
            
%             g1 = a*nz/(2*pi*sigma^2);
%             g2 = exp(-(pn_noz(2)^2)/(2*sigma^2));
%             g3 = exp(-(pn_noz(1)^2)/(2*sigma^2));
            
            gp = g1*g2*g3;
            p_progress(:,p_i) = p_progress(:,p_i)+gp*delta_t;
%             if norm(p_last_layer(:,p_i)-this_p)<26
%                 disp(pn_noz)
%                 disp(gp)
%                 disp('this')
%             end
            
        end
    end
    
    dur = toc;
    T = [T dur];
end

disp(mean(T));
disp(std(T));

% data_result = stlread('result.stl');
data_final = triangulation(data.ConnectivityList, p_progress');
% figure(2); trimesh(data_final,'FaceColor','none','EdgeColor','k'); hold
figure(2); trimesh(data_final,'FaceColor','none','EdgeColor','red'); hold on; 
trimesh(data,'FaceColor','none','EdgeColor','k'); axis equal; view(0,90); hold on;
plot3(all_qs(1,:),all_qs(2,:),all_qs(3,:),'.'); axis equal; hold on;
plot3(all_qs_nz(1,:),all_qs_nz(2,:),all_qs_nz(3,:),'.'); axis equal; hold on;
% plot(q_surface(1,:),q_surface(2,:),'.'); axis equal;
% figure(2); trimesh(data,'FaceColor','none','EdgeColor','k');
title('Final mesh and original surface. (cross section)');
legend('Final mesh','Original surface','Location','bestoutside')
view(0,90);
figure(3); plot(T); title('Deposition simulation time after every motion');
figure(4); plot(all_v); title('Velocity Plot');

function result = impact_angle(nz,nl,margin)
    result = [];
    for i=1:length(nl)
        result = [result 0];
        % check every norm of this point
        for j=1:length(nl{i}(1,:))
            d = dot(nl{i}(:,j),nz);
            if d>cos(deg2rad(margin))
                result(end) = 1;
                break;
            end
        end
    end
end

function nl = point_norm(points,group)
    
    nl = {};
    
    for i=1:length(group)
        v1 = points(:,group(i,2))-points(:,group(i,1));
        v2 = points(:,group(i,3))-points(:,group(i,1));
        vn = cross(v1,v2);
        vn = vn/norm(vn);
        
        for j=1:3
            if length(nl)<group(i,j)
                nl{group(i,j)} = vn;
            else
                if isempty(nl{group(i,j)})
                    nl{group(i,j)} = vn;
                else
                    nl{group(i,j)} = [nl{group(i,j)} vn];
                end
            end
        end
    end
end

function [R,p] = fwd(robot,q)
    [R,p] = fwdkin(robot,q);
    T = [R p; 0 0 0 1]*robot.tool;
    R = T(1:3,1:3);
    p = T(1:3,4);
end
% initialization
clear all; close all;

% stl_file = 'test_stl_compare.stl';
stl_file = 'test_stl_dense.stl';

gm = importGeometry(stl_file);

data = stlread(stl_file);
figure(100); trimesh(data,'FaceColor','none','EdgeColor','k'); hold; 
plotTransforms([0 0 0],[1 0 0 0],'Framesize',10); axis equal;
title("The mold (mesh)")

%data2 = stlread('generic_fan_blade.stl');
%trimesh(data2,'FaceColor','none','EdgeColor','k')

% cold spray simulation, original resolution from stl (low)

% parameters
a = 18;
sigma = 1.4;
stand_off = 25;
v0 = 400;
% v0 = 4;
line_space = 0.5;

% generate robot path
path_ch = 2;

if path_ch == 1 
    % robot path 1
    % one moveL on the top of the mold from z=0 to z=25
    % find the points with max y values (the "top" of the mold)
    max_y = max(data.Points(:,2));
    max_yid = find(data.Points(:,2)==max_y);
    q_start = [mean(data.Points(max_yid,1));max_y;0];
    % q_start = [min(data.Points(max_yid,1));max_y;0];
    Rz_start = [0;1;0];
    q_end = [mean(data.Points(max_yid,1));max_y;25];
    % q_end = [min(data.Points(max_yid,1));max_y;25];
    Rs = [Rz_start];
    q_surface = [q_start q_end];
    qs = [q_start+Rz_start*stand_off q_end+Rz_start*stand_off];

else
    % robot path 2
    % 50+4 moveL with linespacing in between. The lines on the top of the
    % mold have spacing with 150/4 degree.
    mold_r = 0.5; mold_arc_center = [7.5;(7.5*cos(deg2rad(15))-0.5)/sin(deg2rad(15))]; mold_ang = deg2rad(150);
    mold_z = 25; 
    q_num = 50;
    q1 = mold_arc_center+[mold_r*cos(mold_ang/3+deg2rad(15));mold_r*sin(mold_ang/3+deg2rad(15))];
    q2 = mold_arc_center+[mold_r*cos(deg2rad(15));mold_r*sin(deg2rad(15))];
    q_surface = [q1 q2];
    for i=1:q_num
        q_surface = [q_surface q_surface(:,end)+[line_space*cos(deg2rad(-75));line_space*sin(deg2rad(-75))]];
    end
    q_surface_half = q_surface - [2*abs(q_surface(1,:)-(7.5)); zeros(1,length(q_surface(2,:)))];
    q_surface = [flip(q_surface_half,2) q_surface];

    r1 = (q1-mold_arc_center)/norm(q1-mold_arc_center); r2 = (q2-mold_arc_center)/norm(q2-mold_arc_center);
    r_surface = [r1 r2];
    for i=1:q_num
        r_surface = [r_surface r_surface(:,end)];
    end
    r_surface_half = [-r_surface(1,:);r_surface(2,:)];
    r_surface = [flip(r_surface_half,2) r_surface];

    push_off = 1.4*6;
    qs=[]; Rs=[];
    for i=1:2:length(q_surface(1,:))
        qs = [qs [q_surface(:,i)+r_surface(:,i)*stand_off; mold_z+push_off] [q_surface(:,i)+r_surface(:,i)*stand_off; -push_off]]; % back
        Rs = [Rs [r_surface(:,i);0] [r_surface(:,i);0]];
        qs = [qs [q_surface(:,i+1)+r_surface(:,i+1)*stand_off; -push_off] [q_surface(:,i+1)+r_surface(:,i+1)*stand_off; mold_z+push_off]]; % forth
        Rs = [Rs [r_surface(:,i+1);0] [r_surface(:,i+1);0]];
    end
end

writematrix(qs,'qs.csv');

% plot robot waypoint
figure(1); 
trimesh(data,'FaceColor','none','EdgeColor','k'); hold
plot(qs(1,:),qs(2,:),'.'); axis equal;
plot(q_surface(1,:),q_surface(2,:),'.'); axis equal
legend('Mold mesh','Robot path','Projected points');
title("The mold mesh model with robot tool path and projected points (cross section).");
view(0,90);  
    
% Start cold spray deposition

p_init = data.Points';
p_progress = p_init;
p_last_layer = p_progress;
layer_end = true; % if layer end, update the point position and thus the gradient

T = [];
for n=1:length(qs(1,:))-1
    tic;
    nz = Rs(:,n);
    nx = (qs(:,n+1)-qs(:,n)-((qs(:,n+1)-qs(:,n))'*nz)*nz);
    nx = nx/norm(nx);
    ny = cross(nz,nx);
    ln = dot(nx,(qs(:,n+1)-qs(:,n)));
    delta_t = norm(qs(:,n+1)-qs(:,n))/v0;
    
    if layer_end
        p_last_layer = p_progress;
        nl = point_norm(p_last_layer,data.ConnectivityList);
        layer_end = false;
    end
    
    im_ang = impact_angle(nz,nl);
    
    for p_i = 1:length(im_ang)
        if im_ang(p_i) == 0
            continue
        else
            pn_noz = [nx';ny';nz';]*(p_last_layer(:,p_i)-qs(:,n));
            g1 = a*nz/(2*sqrt(2)*pi*sigma);
            g2 = exp(-(pn_noz(2)^2)/(2*sigma^2));
            g3 = erf(pn_noz(1)/(sqrt(2)*sigma))-erf((pn_noz(1)-ln)/(sqrt(2)*sigma));
            gp = g1*g2*g3;
            
            p_progress(:,p_i) = p_progress(:,p_i)+gp*delta_t;
            
            
        end
    end
    
    dur = toc;
    T = [T dur];
end

disp(mean(T));
disp(std(T));

data_final = triangulation(data.ConnectivityList, p_progress');
% figure(2); trimesh(data_final,'FaceColor','none','EdgeColor','k'); hold
figure(2); trimesh(data_final,'FaceColor','none','EdgeColor','red'); hold on; 
trimesh(data,'FaceColor','none','EdgeColor','k'); axis equal; view(0,90);
% plot(q_surface(1,:),q_surface(2,:),'.'); axis equal;
% figure(2); trimesh(data,'FaceColor','none','EdgeColor','k');
title('Final mesh and original surface. (cross section)');
legend('Final mesh','Original surface','Location','bestoutside')
view(0,90);
figure(3); plot(T); title('Deposition simulation time after every motion')

function result = impact_angle(nz,nl)
    result = [];
    for i=1:length(nl)
        result = [result 0];
        % check every norm of this point
        for j=1:length(nl{i}(1,:))
            d = dot(nl{i}(:,j),nz);
            if d>0
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
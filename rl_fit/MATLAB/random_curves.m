clear
load Curve_in_base_frame.csv
num_traj = 200;

X_coord = Curve_in_base_frame(:,1);
Y_coord = Curve_in_base_frame(:,2);
Z_coord = Curve_in_base_frame(:,3);
X_dir = Curve_in_base_frame(:,4);
Y_dir = Curve_in_base_frame(:,5);
Z_dir = Curve_in_base_frame(:,6);

figure(1);
plot3(X_coord,Y_coord,Z_coord,'r.'); grid on;hold on

for i = 1:num_traj

    X_pert = X_coord;
    Y_pert = Y_coord;
    Z_pert = Z_coord;
    N = length(X_coord);

    %==Randomizing the direction of the bump
    Dir = 100*randn(3,1);
    
    %==Randomizing the length of the bump between 0.5 to 0.9 of curve length
    Bump_length = 0.5+0.4*rand;
    Bump_length_idx = floor(Bump_length*N);
    
    %==Randomizing the start of the bump
    Bump_start = (1-Bump_length)*rand;
    Bump_start_idx = ceil(N*Bump_start);
    
    M = 1:Bump_length_idx;
    M = M';
    M = -1+(M/(Bump_length_idx+1))*2;
    Bump = exp(-1./(1-M.^2));
    
    X_pert(Bump_start_idx:Bump_start_idx+Bump_length_idx-1) = X_pert(Bump_start_idx:Bump_start_idx+Bump_length_idx-1) + Dir(1)*Bump;
    Y_pert(Bump_start_idx:Bump_start_idx+Bump_length_idx-1) = Y_pert(Bump_start_idx:Bump_start_idx+Bump_length_idx-1) + Dir(2)*Bump;
    Z_pert(Bump_start_idx:Bump_start_idx+Bump_length_idx-1) = Z_pert(Bump_start_idx:Bump_start_idx+Bump_length_idx-1) + Dir(2)*Bump;

    traj_pert = [X_pert Y_pert Z_pert X_dir Y_dir Z_dir];
    csv_name = ['../data/base/traj_' num2str(i) '.csv'];
    writematrix(traj_pert, csv_name);
    
    plot3(X_pert,Y_pert,Z_pert,'b.');


end




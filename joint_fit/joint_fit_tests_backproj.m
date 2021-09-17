% path=fileparts(which('pwlfmd.py'));
% if count(py.sys.path,path)==0
%     insert(py.sys.path,int32(0),path);
% end
% 
% curve_js = readmatrix('data/from_interp/Curve_backproj_js.csv');
% my_pwlf=py.pwlfmd.MDFit(linspace(1,length(curve_js),length(curve_js)),curve_js)

path=fileparts(which('joint_fit_tests_backproj.py'));
if count(py.sys.path,path)==0
    insert(py.sys.path,int32(0),path);
end
temp=readmatrix('data/from_interp/Curve_in_base_frame.csv');
curve=temp(1:end,1:3);
curve_direction=temp(1:end,4:6);
curve_js = readmatrix('data/from_interp/Curve_backproj_js.csv');
% thresholds=[5.00E-01,1.00E-01,5.00E-02,5.00E-03,5.00E-04,5.00E-05,5.00E-06,5.00E-07];
thresholds=[5.00E-07,1.00E-0];
fit_results=py.joint_fit_tests_backproj.fit_test(py.numpy.array(curve),py.numpy.array(curve_js),py.numpy.array(thresholds));

% fit_results{1}
results_num_breakpoints=double(fit_results{1})';
results_max_cartesian_error=double(fit_results{2})';
results_max_cartesian_error_index=double(fit_results{3})';
results_avg_cartesian_error=double(fit_results{4})';
results_max_orientation_error=double(fit_results{5})';


%%%unable to write header yet
% csv_output=[results_num_breakpoints,results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error];
% writematrix(csv_output,'results/from_interp/cartesian_fit_results.csv')

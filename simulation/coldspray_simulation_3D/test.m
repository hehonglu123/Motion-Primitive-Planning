% initialization
clear all; close all;

origin_file = 'test_stl_compare.stl';
final_file = 'result.stl';

data1 = stlread(origin_file);
data2 = stlread(final_file);


figure(1); trimesh(data2,'FaceColor','none','EdgeColor','red'); hold on; 
trimesh(data1,'FaceColor','none','EdgeColor','k'); axis equal; view(0,90); hold on;
title('Final mesh and original surface. (cross section)');
legend('Final mesh','Original surface','Location','bestoutside')
view(0,90);
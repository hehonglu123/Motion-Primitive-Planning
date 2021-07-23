A = readmatrix('Curve.csv');
curve=A(1:end,1:3);

distance=[0];
for i=2:length(curve)
    distance=[distance;norm(curve(i,:)-curve(i-1,:))];
end
%%%equally distributed index based on distance
x=[0];
for i=2:length(curve)
    x=[x;x(end)+distance(i)/sum(distance)];
end
%%%define interp index
xq=linspace(0,1,1000);
curve_interp = interp1(x,curve,xq,'spline');
plot3(curve_interp(:,1),curve_interp(:,2),curve_interp(:,3))
writematrix(curve_interp,'Curve_interp.csv')
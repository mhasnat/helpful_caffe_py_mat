function spread_3D_points_line(X, label, wtCl)
label = label+1;
% Plot samples of different labels with different colors.
[d,n] = size(X);
if nargin == 1
    label = ones(n,1);
end
assert(n == length(label));

color = 'brgmcyk';

m = length(color);
c = max(label);

view(3);
for i = 1:c
    idc = label==i;
    scatter3(X(1,idc),X(2,idc),X(3,idc),50,color(mod(i-1,m)+1), '.'); hold on;
    vv = [zeros(1,3); wtCl(i, :)];
    plot3(vv(:,1),vv(:,2),vv(:,3), [color(mod(i-1,m)+1) '-'], 'LineWidth', 4); hold on;
end
hold on; [x,y,z] = sphere; 
surface(x,y,z,'FaceColor', 'none'); hold off;

xlabel('dim-1', 'FontSize', 20)
ylabel('dim-2', 'FontSize', 20)
zlabel('dim-3', 'FontSize', 20)
axis equal
grid on
hold off
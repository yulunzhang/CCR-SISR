function Cal_ratio()
clear all; close all; clc
% set parameters
folder_current =pwd;
line_width = 2.5;
marker_size = 5.5;
font_size_graduation = 12;
font_size_axis = 15;
save_form = '-dpng';
fig_form = 'png';

loaddata =  load('conf_Kmeans_pp_1024_finalx3.mat');
num_per_cluster = loaddata.conf_Kmeans_pp.num_per_cluster;

r_all = cumsum(num_per_cluster)./sum(num_per_cluster);
figure;
cluster_label = 1:1:length(num_per_cluster);
plot(cluster_label, r_all, 'LineWidth', line_width);
xlabel('\bfIndex of Cluster Center', 'FontSize', font_size_axis);
ylabel('\bfCenter Preserving Ratio', 'FontSize', font_size_axis);
set(gca,'fontsize',font_size_graduation,'fontweight','bold');
set(gca, 'LineWidth', 2);
axis([0 1100 0 1]);
% hold on
% axes('linewidth', 2, 'box', 'on');
end
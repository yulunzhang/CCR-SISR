function reduce_clusters()
clear all; close all; clc
loaddata = load('conf_Kmeans_pp_2048_finalx3.mat');
num_per_cluster = loaddata.conf_Kmeans_pp.num_per_cluster;
num_patch_whole = sum(num_per_cluster);
num_cluster = length(num_per_cluster);
for i = 1:num_cluster
    fprintf('i=%d, percentage=%f\n', i, sum(num_per_cluster(1:i))/num_patch_whole);
end
end
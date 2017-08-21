function compare_sr_methods()
% by Yulun Zhang
% zhangyl14@mails.tsinghua.edu.cn
% updated on 2015/03/08
% we conduct SR methods: A+, KMSE, 
clear all; close all; clc

p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm

upscaling = 3;
num_clusters = 1024;
num_nn_centers = 40;
num_nn_patches = 2048;
%
input_dir = 'Set_dong';
pattern = '*.tif';

%% K-means clustering; dist = Euc 
mat_file = ['conf_kmeans_pp_' num2str(num_clusters) '_x' num2str(upscaling) '.mat'];
if exist(mat_file, 'file')
    disp(['Loading clustered centers...' mat_file]);
    load(mat_file, 'conf_kmeans_pp');
else
    disp(['Number of clusters is ' num2str(num_clusters) ' using K-means plus plus...']);
    % Simulatino settings
    conf_kmeans_pp.scale = upscaling;
    conf_kmeans_pp.level = 1;
    conf_kmeans_pp.window = [3 3];
    conf_kmeans_pp.border = [1 1];
    
    conf_kmeans_pp.upsample_factor = upscaling;
    O = zeros(1, conf_kmeans_pp.upsample_factor-1);
    G = [1 O -1];
    L = [1 O -2 O 1]/2;
    conf_kmeans_pp.filters = {G, G.', L, L.'};
    conf_kmeans_pp.interpolate_kernel = 'bicubic';
    conf_kmeans_pp.overlap = [1 1];
    if upscaling <= 2
        conf_kmeans_pp.overlap = [1 1];
    end
    startt = tic;
    conf_kmeans_pp = cluster_kmeans_pp(conf_kmeans_pp, load_images(glob('CVPR08-SR/Data/Training', '*.bmp')), num_clusters);
    conf_kmeans_pp.overlap = conf_kmeans_pp.window - [1 1];
    conf_kmeans_pp.trainingtime = toc(startt);
    toc(startt)
    save(mat_file, 'conf_kmeans_pp');
    
end







end
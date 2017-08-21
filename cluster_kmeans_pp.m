function [conf] = cluster_kmeans_pp(conf, hires, dictsize)
% Sample patches (from high-res. images) and extract features (from low-res.)
% for the Super Resolution algorithm training phase, using specified scale 
% factor between high-res. and low-res.

% Load training high-res. image set and resample it
hires = modcrop(hires, conf.scale); % crop a bit (to simplify scaling issues)
% Scale down images
lores = resize(hires, 1/conf.scale, conf.interpolate_kernel);

midres = resize(lores, conf.upsample_factor, conf.interpolate_kernel);
features = collect(conf, midres, conf.upsample_factor, conf.filters);
clear midres

interpolated = resize(lores, conf.scale, conf.interpolate_kernel);
clear lores
patches = cell(size(hires));
for i = 1:numel(patches) % Remove low frequencies
    patches{i} = hires{i} - interpolated{i};
end
clear hires interpolated

patches = collect(conf, patches, conf.scale, {});

% Set KSVD configuration
%ksvd_conf.iternum = 20; % TBD
ksvd_conf.iternum = 20; % TBD
ksvd_conf.memusage = 'high'; % higher usage doesn't fit...
%ksvd_conf.dictsize = 5000; % TBD
ksvd_conf.dictsize = dictsize; % TBD
ksvd_conf.Tdata = 3; % maximal sparsity: TBD
ksvd_conf.samples = size(patches,2);

% PCA dimensionality reduction
C = double(features * features');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix 
D = cumsum(D) / sum(D);
k = find(D >= 1e-3, 1); % ignore 0.1% energy
conf.V_pca = V(:, k:end); % choose the largest eigenvectors' projection
conf.ksvd_conf = ksvd_conf;
features_pca = conf.V_pca' * features;

% Combine into one large training set
clear C D V
ksvd_conf.data = double(features_pca);
clear features_pca

% Clustering by K-means plus plus
folder_current = pwd;
run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);%Vl_feat library for k-nearest neighbor queries
[centers, assignments] = vl_kmeans(ksvd_conf.data, dictsize, 'Initialization', 'plusplus');
% sort
arr_train_feature = hist(double(assignments), dictsize);
[num_per_cluster, IX] = sort(arr_train_feature, 'descend');
cluster_centers = centers(:, IX);

conf.dict_lore_kmeans = cluster_centers;
conf.num_per_cluster = num_per_cluster;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

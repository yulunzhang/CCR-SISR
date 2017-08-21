function [conf] = learn_dict_kmeans(conf, hires, dictsize)
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
% Training process (will take a while)
% tic;
% fprintf('Training [%d x %d] dictionary on %d vectors using K-SVD\n', ...
%     size(ksvd_conf.data, 1), ksvd_conf.dictsize, size(ksvd_conf.data, 2))
% [conf.dict_lores, gamma, err] = ksvd(ksvd_conf); 
% toc;
% conf.ksvd_conf.err = err;
% % X_lores = dict_lores * gamma
% % X_hires = dict_hires * gamma {hopefully}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [plores ~] = collectSamplesScales(conf, load_images(...            
%         glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98); 
% fprintf('Training [%d x %d] dictionary on %d vectors using K-means\n', ...
%     size(ksvd_conf.data, 1), ksvd_conf.dictsize, size(ksvd_conf.data, 2))
% K-means clustering parameters setting
seed = RandStream('mcg16807', 'Seed', 0);
RandStream.setGlobalStream(seed)
num_iteration = 100;
num_cluster = dictsize;
opts = statset('Display', 'iter', 'MaxIter', num_iteration);
data_to_train_cluster = ksvd_conf.data';

[IDX, C] = kmeans(data_to_train_cluster, num_cluster, 'emptyaction', 'drop', 'options', opts);
% sort
arr_train_feature = hist(IDX, num_cluster);
[arr_train_feature_sort, IX] = sort(arr_train_feature, 'descend');
cluster_centers = C(IX, :);
conf.dict_lore_kmeans = cluster_centers';
conf.num_per_cluster_1 = arr_train_feature_sort;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

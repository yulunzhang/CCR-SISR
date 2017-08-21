function PP_lambda()
clear all; close all; clc
upscaling = 2; % the magnification factor x2, x3, x4...
dict_size = 1024;
lambda = 0;
p = pwd;
addpath(genpath(fullfile(p, '/ksvdbox_Bo'))) % MI-K-SVD dictionary training algorithm
rmpath(fullfile(p, '/ksvdbox'))
rmpath(fullfile(p, '/ompbox'))
for idx_lambda = 1:100
    mat_file = ['MIKSVD_' num2str(dict_size) '_x' num2str(upscaling) '_lambda' num2str(lambda) '.mat'];    
    disp(['Training dictionary of size ' num2str(dict_size) ' using MI-KSVD approach...']);

    % Simulation settings
    conf_MIKSVD.scale = upscaling; % scale-up factor
    conf_MIKSVD.level = 1; % # of scale-ups to perform
    conf_MIKSVD.window = [3 3]; % low-res. window size
    conf_MIKSVD.border = [1 1]; % border of the image (to ignore)

    % High-pass filters for feature extraction (defined for upsampled low-res.)
    conf_MIKSVD.upsample_factor = upscaling; % upsample low-res. into mid-res.
    O = zeros(1, conf_MIKSVD.upsample_factor-1);
    G = [1 O -1]; % Gradient
    L = [1 O -2 O 1]/2; % Laplacian
    conf_MIKSVD.filters = {G, G.', L, L.'}; % 2D versions
    conf_MIKSVD.interpolate_kernel = 'bicubic';

    conf_MIKSVD.overlap = [1 1]; % partial overlap (for faster training)
    if upscaling <= 2
        conf_MIKSVD.overlap = [1 1]; % partial overlap (for faster training)
    end

    startt = tic;
    conf_MIKSVD = learn_dict_MIKSVD(conf_MIKSVD, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp') ...
        ), dict_size, lambda);       
    conf_MIKSVD.overlap = conf_MIKSVD.window - [1 1]; % full overlap scheme (for better reconstruction)    
    conf_MIKSVD.trainingtime = toc(startt);
    toc(startt)

    save(mat_file, 'conf_MIKSVD');  
    Dl_miksvd = conf_MIKSVD.dict_lores_MIKSVD;
    amc_all(idx_lambda) = (sum(sum(abs(Dl_miksvd'*Dl_miksvd)))-sum(diag(Dl_miksvd'*Dl_miksvd)))/(dict_size*(dict_size-1));
    
    lambda = lambda + 0.01;
end
save('AMC.mat', 'amc_all');
rmpath(fullfile(p, '/ksvdbox_Bo'))
% train call        
end

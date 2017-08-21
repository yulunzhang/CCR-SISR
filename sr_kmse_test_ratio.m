function sr_kmse_test_ratio()
% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution
% Example code
%
% March 22, 2013. Radu Timofte, VISICS @ KU Leuven
%
% Revised version: (includes all [1] methods)
% October 3, 2013. Radu Timofte, CVL @ ETH Zurich
%
% Updated version: (adds A+ methods [2])
% September 5, 2014. Radu Timofte, CVL @ ETH Zurich
% %
% Please reference to both:
% [1] Radu Timofte, Vincent De Smet, Luc Van Gool.
% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution.
% International Conference on Computer Vision (ICCV), 2013. 
%
% [2] Radu Timofte, Vincent De Smet, Luc Van Gool.
% A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution.
% Asian Conference on Computer Vision (ACCV), 2014. 
%
% For any questions, email me by timofter@vision.ee.ethz.ch
%

clear all; close all; clc;  
warning off    
p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm
addpath('metrix_mux');


imgscale = 1;   % the scale reference we work with
flag = 0;       % flag = 0 - only GR, ANR, A+, and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied

upscaling = 3; % the magnification factor x2, x3, x4...

input_dir = 'Set30'; % Directory with input images from Set5 image dataset
%input_dir = 'Set14'; % Directory with input images from Set14 image dataset

% pattern = '*.bmp'; % Pattern to process
pattern = '*.bmp'; % Pattern to process

dict_sizes = [2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];

%nn_patch_max = 1;
%d = 7
%for nn=1:28
%nn= 28

clusterszA = 2048; % neighborhood size for A+
clusterszApp = 2048; % neighborhood size for A++

num_patches_cluster = 5000000;
%maxIter = 20;                   % if 0, do not use backprojection

disp('The experiment corresponds to the results from Table 2 in the referenced [1] and [2] papers.');

disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling) '.']);
if flag==1
    disp('All methods are employed : Bicubic, Yang et al., Zeyde et al., GR, ANR, NE+LS, NE+NNLS, NE+LLE, A+ (0.5 mil), A+, A+ (16 atoms), App_Zhang, KMSE');    
else
    disp('We run only for Bicubic, GR, ANR , A+ and A++ methods, the other get the Bicubic result by default.');
end

fprintf('\n\n');
for num_centers_use = 6
    
for d = 10   %1024
    %d=17; %65536
    %d=16; %32768
    %d=15; %16384
    %d=14; %16384
    %d=13; %8192
    %d=12; %4096
    %d=11; %2048
    %d=10; %1024
    %d = 9; %512
    %d = 8; %256
    %d = 7; %128
    %d = 6; % 64
    %d = 5; % 32
    %d=4;  %16
    %d=3;  %8
    %d=2; %4
    %d=1; %2
    
    %num_centers_use = 6;
    llambda_kmeans_sub = 0.03;
    idx_nn_patch_max = 11;
    nn_patch_max = 2^idx_nn_patch_max;
    disp(['nn_patch_max = ' num2str(nn_patch_max)]);
    
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    disp(['Upscaling x' num2str(upscaling) ' ' input_dir ' with Zeyde dictionary of size = ' num2str(dict_sizes(d))]);
    
    mat_file = ['conf_Zeyde_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf');

    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';

        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf = learn_dict(conf, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp') ...
            ), dict_sizes(d));       
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf.trainingtime = toc(startt);
        toc(startt)        
        save(mat_file, 'conf');
        % train call        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mat_file = ['conf_MIKSVD_' num2str(dict_sizes(d)) '_x' num2str(upscaling) '_lambda003']; 
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_MIKSVD');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using MI-KSVD approach...']);
        
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
        conf_MIKSVD = learn_dict_MIKSVD(conf_MIKSVD, load_images(glob('CVPR08-SR/Data/Training', '*.bmp')), dict_sizes(d), 0.03);       
        conf_MIKSVD.overlap = conf_MIKSVD.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_MIKSVD.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf_MIKSVD');                       
        
        % train call        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%  Dictionary learning via K-means clustering  %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_file = ['conf_Kmeans_pp_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_Kmeans_pp');

    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using K-means approach...']);
        % Simulation settings
        conf_Kmeans_pp.scale = upscaling; % scale-up factor
        conf_Kmeans_pp.level = 1; % # of scale-ups to perform
        conf_Kmeans_pp.window = [3 3]; % low-res. window size
        conf_Kmeans_pp.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_Kmeans_pp.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_Kmeans_pp.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_Kmeans_pp.filters = {G, G.', L, L.'}; % 2D versions
        conf_Kmeans_pp.interpolate_kernel = 'bicubic';

        conf_Kmeans_pp.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_Kmeans_pp.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf_Kmeans_pp = cluster_kmeans_pp(conf_Kmeans_pp, load_images(glob('CVPR08-SR/Data/Training', '*.bmp')), dict_sizes(d));       
        conf_Kmeans_pp.overlap = conf_Kmeans_pp.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_Kmeans_pp.trainingtime = toc(startt);
        toc(startt)        
        save(mat_file, 'conf_Kmeans_pp');
        % train call        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end
    
    %% GR
    if dict_sizes(d) < 10000
        conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';    
        conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
    else
        % here should be an approximation
        conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
        conf.ProjM = [];
    end
    
    conf.filenames = glob(input_dir, pattern); % Cell array      
    
%     conf.desc = {'Original', 'Bicubic', 'Yang et al.', ...
%         'Zeyde et al.', 'Our GR', 'Our ANR', ...
%         'NE+LS','NE+NNLS','NE+LLE','Our A+ (0.5mil)','Our A+', 'Our A+ (16atoms)'};
    conf.desc = {'Gnd', 'Bicubic', 'Yang', 'Zeyde', 'GR', 'ANR', 'NE_LS','NE_NNLS','NE_LLE','Aplus05mil','Aplus', 'Aplus16atoms', 'App_Zhang', 'KMSE', 'KMSE_sub', 'KMSE_10', 'KMSE_15', 'KMSE_20', 'KMSE_25', 'KMSE_30', 'KMSE_35', 'KMSE_40', 'KMSE_45', 'KMSE_50', 'KMSE_55', 'KMSE_60', 'KMSE_65', 'KMSE_70', 'KMSE_75', 'KMSE_80', 'KMSE_85', 'KMSE_90', 'KMSE_95'};
    conf.results = {};
    
    %conf.points = [1:10:size(conf.dict_lores,2)];
    conf.points = [1:1:size(conf.dict_lores,2)];
    
    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    
    % precompute for ANR the anchored neighborhoods and the projection matrices for
    % the dictionary 
    
    conf.PPs = [];    
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end
    D = abs(conf.pointslo'*conf.dict_lores);    
    
    for i = 1:length(conf.points)
        [vals idx] = sort(D(i,:), 'descend');
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));        
            conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';    
        end
    end    
    
    ANR_PPs = conf.PPs; % store the ANR regressors
    
    save([tag '_' mat_file '_ANR_projections_imgscale_' num2str(imgscale)],'conf');
    
    %% A+ computing the regressors
    Aplus_PPs = [];
        
    fname = ['Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_5mil.mat'];
    
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute A+ regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  

        if size(plores,2) > 5000000                
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda = 0.1;

        for i = 1:size(conf.dict_lores,2)
            D = pdist2(single(plores'),single(conf.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            Aplus_PPs{i} = Hi*inv(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
%Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(fname,'Aplus_PPs','ttime', 'number_samples');   
        toc
    end    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% A++ computing the regressors
    App_PPs = [];
        
    fname = ['App_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszApp) 'nn_5mil_lambda003.mat'];
    
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute A++ regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  

        if size(plores,2) > num_patches_cluster                
            plores = plores(:,1:num_patches_cluster);
            phires = phires(:,1:num_patches_cluster);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda_App = 0.1;

        for i = 1:size(conf.dict_lores,2)
            D = abs(single(plores')*single(conf_MIKSVD.dict_lores_MIKSVD(:,i)));
            [~, idx] = sort(D, 'descend');                
            Lo = plores(:, idx(1:clusterszApp));                                    
            Hi = phires(:, idx(1:clusterszApp));
            App_PPs{i} = Hi*inv(Lo'*Lo+llambda_App*eye(size(Lo,2)))*Lo'; 
            %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(fname,'App_PPs','ttime', 'number_samples');   
        toc
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%  KMSE computing the regressors  %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    kmeans_pp_PPs = [];
        
    fname = ['kmeans_pp_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszApp) 'nn_5mil.mat'];
    
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute kmeans_pp regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  

        if size(plores,2) > num_patches_cluster                
            plores = plores(:,1:num_patches_cluster);
            phires = phires(:,1:num_patches_cluster);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda_kmeans_pp = 0.1;

        for i = 1:size(conf.dict_lores,2)
            D = pdist2(single(plores'),single(conf_Kmeans_pp.dict_lore_kmeans(:,i)'));
            [~, idx] = sort(D, 'ascend');                
            Lo = plores(:, idx(1:clusterszApp));                                    
            Hi = phires(:, idx(1:clusterszApp));
            kmeans_pp_PPs{i} = Hi*inv(Lo'*Lo+llambda_kmeans_pp*eye(size(Lo,2)))*Lo'; 
            %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(fname,'kmeans_pp_PPs','ttime', 'number_samples');   
        toc
    end 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%  KMSE_sub computing the regressors  %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    kmeans_sub_PPs = [];
    %num_centers_use = 30;       % num_centers_use nearest centers to use 
    %nn_patch_max = 2048;        % max number of the nn patches to each cluster center
    fname = ['kmeans_sub_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(nn_patch_max) 'npm_' num2str(num_centers_use) 'sub_whole_lambda' num2str(llambda_kmeans_sub*1000) '.mat'];
    %fname = ['kmeans_sub_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(nn_patch_max) 'nn_' num2str(num_centers_use) 'sub_whole.mat'];
    % e.g. kmeans_sub_x3_1024atoms_2048nn_40sub_whole.mat
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute kmeans_sub regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  

%         if size(plores,2) > num_patches_cluster                
%             plores = plores(:,1:num_patches_cluster);
%             phires = phires(:,1:num_patches_cluster);
%         end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        %llambda_kmeans_sub = 0.1;
        %cluster the whole data with kmeans plus plus
        folder_current = pwd;
        run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);
        [centers, assignments] = vl_kmeans(plores, dict_sizes(10), 'Initialization', 'plusplus');
        assignments = double(assignments);
        %load('cluster_x3_whole.mat');
        for i = 1:size(conf.dict_lores, 2)
            D = pdist2(single(centers'), single(conf_Kmeans_pp.dict_lore_kmeans(:, i)'));
            [~, idx_centers] = sort(D, 'ascend');
            
            idx_centers_use = idx_centers(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(assignments == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = plores(:, idx_patch_use);
            sub_phires = phires(:, idx_patch_use);
            
            if nn_patch_max <= length(idx_patch_use)
                sub_D = pdist2(single(sub_plores'), single(conf_Kmeans_pp.dict_lore_kmeans(:, i)'));
                [~, sub_idx] = sort(sub_D, 'ascend');
                Lo = sub_plores(:, sub_idx(1:nn_patch_max));
                Hi = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                Lo = sub_plores;
                Hi = sub_phires;
            end
              
           %kmeans_sub_PPs{i} = Hi*inv(Lo'*Lo+llambda_kmeans_sub*eye(size(Lo,2)))*Lo';
           kmeans_sub_PPs{i} = Hi*((Lo'*Lo+llambda_kmeans_sub*eye(size(Lo,2)))\Lo');
        end
        
     
        clear plores
        clear phires
        clear sub_plores
        clear sub_phires
        
        ttime = toc(ttime);        
        save(fname,'kmeans_sub_PPs','ttime', 'number_samples');   
        toc
    end 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%  KMSE_abs computing the regressors  %%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     KMSE_abs_PPs = [];
%     loaddata_abs = load('cluster_abs_x3.mat');    
%     fname = ['KMSE_abs_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszApp) 'nn_5mil.mat'];
%     
%     if exist(fname,'file')
%        load(fname);
%     else
%         %%
%        disp('Compute KMSE_abs regressors');
%        ttime = tic;
%        tic
%        [plores phires] = collectSamplesScales(conf, load_images(...            
%         glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  
% 
%         if size(plores,2) > num_patches_cluster                
%             plores = plores(:,1:num_patches_cluster);
%             phires = phires(:,1:num_patches_cluster);
%         end
%         number_samples = size(plores,2);
%         
%         % l2 normalize LR patches, and scale the corresponding HR patches
%         l2 = sum(plores.^2).^0.5+eps;
%         l2n = repmat(l2,size(plores,1),1);    
%         l2(l2<0.1) = 1;
%         plores = plores./l2n;
%         phires = phires./repmat(l2,size(phires,1),1);
%         clear l2
%         clear l2n
% 
%         llambda_KMSE_abs = 0.1;
%         
%         for i = 1:size(conf.dict_lores,2)
%             D = abs(single(plores')*single(loaddata_abs.centers(:,i)));
%             [~, idx] = sort(D, 'descend');                  
%             Lo = plores(:, idx(1:clusterszApp));                                    
%             Hi = phires(:, idx(1:clusterszApp));
%             KMSE_abs_PPs{i} = Hi*inv(Lo'*Lo+llambda_KMSE_abs*eye(size(Lo,2)))*Lo'; 
%             %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
%         end        
%         clear plores
%         clear phires
%         
%         ttime = toc(ttime);        
%         save(fname,'KMSE_abs_PPs','ttime', 'number_samples');   
%         toc
%         
%     end 
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%  KMSE_Euclidean computing the regressors  %%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     KMSE_Euc_PPs = [];
%     loaddata_Euc = load('clusters_Euclidean_x3.mat');    
%     fname = ['KMSE_Euc_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszApp) 'nn_5mil.mat'];
%     
%     if exist(fname,'file')
%        load(fname);
%     else
%         %%
%        disp('Compute KMSE_Euc regressors');
%        ttime = tic;
%        tic
%        [plores phires] = collectSamplesScales(conf, load_images(...            
%         glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  
% 
%         if size(plores,2) > num_patches_cluster                
%             plores = plores(:,1:num_patches_cluster);
%             phires = phires(:,1:num_patches_cluster);
%         end
%         number_samples = size(plores,2);
%         
%         % l2 normalize LR patches, and scale the corresponding HR patches
%         l2 = sum(plores.^2).^0.5+eps;
%         l2n = repmat(l2,size(plores,1),1);    
%         l2(l2<0.1) = 1;
%         plores = plores./l2n;
%         phires = phires./repmat(l2,size(phires,1),1);
%         clear l2
%         clear l2n
% 
%         llambda_KMSE_Euc = 0.1;
%         
%         
%         for i = 1:size(conf.dict_lores,2)
%             D = pdist2(single(plores'),single(loaddata_Euc.centers(:,i)'));
%             [~, idx] = sort(D, 'ascend');                    
%             Lo = plores(:, idx(1:clusterszApp));                                    
%             Hi = phires(:, idx(1:clusterszApp));
%             KMSE_Euc_PPs{i} = Hi*inv(Lo'*Lo+llambda_KMSE_Euc*eye(size(Lo,2)))*Lo'; 
%             %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
%         end        
%         clear plores
%         clear phires
%         
%         ttime = toc(ttime);        
%         save(fname,'KMSE_Euc_PPs','ttime', 'number_samples');   
%         toc
%     end 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% A+ (0.5mil) computing the regressors with 0.5 milion training samples
    Aplus05_PPs = [];    
    
    fname = ['Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_05mil.mat'];    
    
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute A+ (0.5 mil) regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 1,1);  

        if size(plores,2) > 500000                
            plores = plores(:,1:500000);
            phires = phires(:,1:500000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);      
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda = 0.1;

        for i = 1:size(conf.dict_lores,2)
            D = pdist2(single(plores'),single(conf.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            Aplus05_PPs{i} = Hi*inv(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(fname,'Aplus05_PPs','ttime', 'number_samples');   
        toc
    end            
    
    %% load the A+ (16 atoms) for comparison results
    conf16 = [];       
    fname = ['Aplus_x' num2str(upscaling) '_16atoms' num2str(clusterszA) 'nn_05mil.mat'];
    fnamec = ['Set14_x' num2str(upscaling) '_16atoms_conf_Zeyde_16_finalx' num2str(upscaling) '_ANR_projections_imgscale_' num2str(imgscale) '.mat']; 
    if exist(fname,'file') && exist(fnamec,'file')
       kk = load(fnamec);
       conf16 = kk.conf;       
       kk = load(fname);       
       conf16.PPs = kk.Aplus05_PPs;
       clear kk
    end
    %%    
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);
%     conf.result_dir = qmkdir(['Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
%     conf.result_dirRGB = qmkdir(['ResultsRGB-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
%     conf.result_dir = qmkdir(['Results-' sprintf('%s_x%d-', input_dir, upscaling) datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
%     conf.result_dirRGB = qmkdir(['ResultsRGB-' sprintf('%s_x%d-', input_dir, upscaling) datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    conf.result_dir = qmkdir(['Res-' sprintf('%s_x%d-', input_dir, upscaling) 'd' num2str(dict_sizes(d)) '_ncu' num2str(num_centers_use) '_npm' num2str(nn_patch_max) 'lam' num2str(llambda_kmeans_sub*1000)]);
    conf.result_dirRGB = qmkdir(['ResRGB-' sprintf('%s_x%d-', input_dir, upscaling) 'd' num2str(dict_sizes(d)) '_ncu' num2str(num_centers_use) '_npm' num2str(nn_patch_max) 'lam' num2str(llambda_kmeans_sub*1000)]);
%     conf.result_dir = qmkdir(['Results-' sprintf('%s_x%d', input_dir, upscaling)]);
%     conf.result_dirRGB = qmkdir(['ResultsRGB-' sprintf('%s_x%d', input_dir, upscaling)]);
    %%
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    res =[];
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f}); 
        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);

            low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
            end
            
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);    
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);    
        end
        
        res{1} = interpolated;
                        
        if (flag == 1) && (dict_sizes(d) == 1024) && (upscaling==3)
            startt = tic;
            res{2} = {yima(low{1}, upscaling)};                        
            toc(startt)
            conf.countedtime(2,i) = toc(startt);
        else
            res{2} = interpolated;
        end
        
        if (flag == 1)
            startt = tic;
            res{3} = scaleup_Zeyde(conf, low);
            toc(startt)
            conf.countedtime(3,i) = toc(startt);    
        else
            res{3} = interpolated;
        end
        
        if flag == 1
            startt = tic;
            res{4} = scaleup_GR(conf, low);
            toc(startt)
            conf.countedtime(4,i) = toc(startt);    
        else
            res{4} = interpolated;
        end
        
        startt = tic;
        conf.PPs = ANR_PPs;
        res{5} = scaleup_ANR(conf, low);
        toc(startt)
        conf.countedtime(5,i) = toc(startt);    
        
        if flag == 1
            startt = tic;
            if 12 < dict_sizes(d)
                res{6} = scaleup_NE_LS(conf, low, 12);
            else
                res{6} = scaleup_NE_LS(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(6,i) = toc(startt);    
        else
            res{6} = interpolated;
        end
        
        if flag == 1
            startt = tic;
            if 24 < dict_sizes(d)
                res{7} = scaleup_NE_NNLS(conf, low, 24);
            else
                res{7} = scaleup_NE_NNLS(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(7,i) = toc(startt);    
        else
            res{7} = interpolated;
        end
        
        if flag == 1
            startt = tic;
            if 24 < dict_sizes(d)
                res{8} = scaleup_NE_LLE(conf, low, 24);
            else
                res{8} = scaleup_NE_LLE(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(8,i) = toc(startt);    
        else
            res{8} = interpolated;
        end
            
        % A+ (0.5 mil)
        if flag == 1 && ~isempty(Aplus05_PPs)
            fprintf('A+ (0.5mil)\n');
            conf.PPs = Aplus05_PPs;
            startt = tic;
            res{9} = scaleup_ANR(conf, low);
            toc(startt)
            conf.countedtime(9,i) = toc(startt);    
        else
            res{9} = interpolated;
        end
        
        % A+
        if ~isempty(Aplus_PPs)
            fprintf('A+\n');
            conf.PPs = Aplus_PPs;
            startt = tic;
            res{10} = scaleup_ANR(conf, low);
            toc(startt)
            conf.countedtime(10,i) = toc(startt);    
        else
            res{10} = interpolated;
        end        
        % A+ 16atoms
        if flag == 1 && ~isempty(conf16)
            fprintf('A+ 16atoms\n');
            startt = tic;
            res{11} = scaleup_ANR(conf16, low);
            toc(startt)
            conf.countedtime(11,i) = toc(startt);    
        else
            res{11} = interpolated;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~isempty(App_PPs)
            fprintf('A++\n');
            conf.PPs = App_PPs;
            startt = tic;
            
            conf.pointslo = conf_MIKSVD.dict_lores_MIKSVD(:,conf.points);
            res{12} = scaleup_APP_Zhang_MIKSVD(conf, low);
            toc(startt)
            conf.countedtime(12,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{12} = interpolated;
        end 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%  SR via kmeans_pp  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~isempty(kmeans_pp_PPs)
            fprintf('kmeans_pp\n');
            conf.PPs = kmeans_pp_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,conf.points);
            res{13} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(13,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{13} = interpolated;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%  SR via kmeans_sub  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,conf.points);
            res{14} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(14,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{14} = interpolated;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%  SR via KMSE_10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_10\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:2);
            res{15} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(15,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{15} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_15 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_15\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:4);
            res{16} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(16,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{16} = interpolated;
        end
        
        %%%%%%%%%%%  SR via KMSE_20 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_20\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:8);
            res{17} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(17,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{17} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_25 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_25\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:13);
            res{18} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(18,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{18} = interpolated;
        end
        
        %%%%%%%%%%%  SR via KMSE_30 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_30\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:20);
            res{19} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(19,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{19} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_35 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_35\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:28);
            res{20} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(20,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{20} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_40 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_40\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:40);
            res{21} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(21,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{21} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_45 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_45\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:53);
            res{22} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(22,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{22} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_50 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_50\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:70);
            res{23} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(23,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{23} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_55 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_55\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:90);
            res{24} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(24,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{24} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_60 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_60\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:114);
            res{25} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(25,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{25} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_65 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_65\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:145);
            res{26} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(26,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{26} = interpolated;
        end
        
        %%%%%%%%%%%  SR via KMSE_70 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_70\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:184);
            res{27} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(27,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{27} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_75 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_75\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:233);
            res{28} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(28,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{28} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_80 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_80\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:295);
            res{29} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(29,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{29} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_85 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_85\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:376);
            res{30} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(30,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{30} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_90 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_90\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:485);
            res{31} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(31,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{31} = interpolated;
        end
        %%%%%%%%%%%  SR via KMSE_95 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        if ~isempty(kmeans_sub_PPs)
            fprintf('kmeans_sub_95\n');
            conf.PPs = kmeans_sub_PPs;
            startt = tic;
            
            conf.pointslo = conf_Kmeans_pp.dict_lore_kmeans(:,1:644);
            res{32} = scaleup_KMSE(conf, low);
            toc(startt)
            conf.countedtime(32,i) = toc(startt);
            conf.pointslo = conf.dict_lores(:,conf.points);
        else
            res{32} = interpolated;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1}, res{8}{1}, ...
            res{9}{1}, res{10}{1}, res{11}{1}, res{12}{1}, res{13}{1}, res{14}{1}, res{15}{1}, res{16}{1}, res{17}{1}, res{18}{1}, res{19}{1}, res{20}{1}, res{21}{1}, res{22}{1}, res{23}{1}, res{24}{1}, res{25}{1}, res{26}{1}, res{27}{1}, res{28}{1}, res{29}{1}, res{30}{1}, res{31}{1}, res{32}{1});
        
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)            
%             conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);   
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]); 
            imwrite(result(:, :, j), conf.results{i}{j});

%             conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end
            
            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end        
        conf.filenames{i} = f;
    end   
    conf.duration = cputime - t;

    % Test performance
     
%      scores_PSNR = run_comparison_PSNR(conf);
%      scores_SSIM = run_comparison_SSIM(conf);
%      scores_VIF = run_comparison_VIF(conf);
    %process_scores_Tex(conf, scores,length(conf.filenames));
    % PSNR
    run_comparisonRGB_PSNR(conf); % provides color images and HTML summary
    % SSIM
    run_comparisonRGB_SSIM(conf); 
    % VIF
    run_comparisonRGB_VIF(conf);
    % 
    %%    
    %save([tag '_' mat_file '_results_imgscale_' num2str(imgscale)],'conf','scores');
end
end
%
end
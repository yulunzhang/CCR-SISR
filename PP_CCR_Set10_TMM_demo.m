function PP_CCR_Set10_TMM_demo()
% README for CCR
% Updated on 2016/07/05, by Yulun Zhang, zhangyl14@mails.tsinghua.edu.cn
% 
% Reproduce the results presented in our VCIP2015 best student paper 'Adaptive Local Nonparametric Regression for Fast Single Image Super-Resolution'
% 
% Just run 'PP_CCR_Set10_TMM_demo.m' to get a start.
% 
% This demo code is based on the codes released by Timofte et al.. Many thanks to them!
% 
% Please cite:
% [1] Radu Timofte, Vincent De Smet, Luc Van Gool:
% A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution, ACCV 2014.
% 
% [2] Radu Timofte, Vincent De Smet, Luc Van Gool:
% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution, ICCV 2013.
% 
% [3] Yulun Zhang, Yongbing Zhang, Jian Zhang, and Qionghai Dai:
% CCR: Clustering and collaborative representation for fast single image super-resolution, TMM 2016.
% 
% @article{zhang2015ccr,
%   title={CCR: Clustering and collaborative representation for fast single image super-resolution},
%   author={Zhang, Yulun and Zhang, Yongbing and Zhang, Jian and Wang, Haoqian and Dai, Qiongdai},
%   journal={{IEEE} Trans. Multimedia},
%   volume={18},
%   number={3},
%   pages={405--417},
%   month={Mar.},
%   year={2016},
%   publisher={IEEE}
% }
%
clear all; close all; clc;  
warning off all   
p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm



imgscale = 1;   % the scale reference we work with
flag = 0;       % flag = 0 - only GR, ANR, A+, CCR, and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied

upscaling = 3; % the magnification factor

input_dir = 'Set10TMM'; % Directory with input images

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
    llambda_CCR = 0.03;
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%  Dictionary learning via K-means clustering  %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_file = ['conf_Kmeans_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_Kmeans');

    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using K-means approach...']);
        % Simulation settings
        conf_Kmeans.scale = upscaling; % scale-up factor
        conf_Kmeans.level = 1; % # of scale-ups to perform
        conf_Kmeans.window = [3 3]; % low-res. window size
        conf_Kmeans.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_Kmeans.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_Kmeans.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_Kmeans.filters = {G, G.', L, L.'}; % 2D versions
        conf_Kmeans.interpolate_kernel = 'bicubic';

        conf_Kmeans.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_Kmeans.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf_Kmeans = cluster_kmeans_pp(conf_Kmeans, load_images(glob('CVPR08-SR/Data/Training', '*.bmp')), dict_sizes(d));       
        conf_Kmeans.overlap = conf_Kmeans.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_Kmeans.trainingtime = toc(startt);
        toc(startt)        
        save(mat_file, 'conf_Kmeans');
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
    
    conf.desc = {'Gnd', 'Bicubic', 'Yang', 'Zeyde', 'GR', 'ANR', 'NE_LS','NE_NNLS','NE_LLE', 'Aplus', 'CCR'};
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
            fprintf('Compute A+ regressors: %d/%d\n', i, size(conf.dict_lores,2));
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
    %save([tag '_' mat_file '_ANR_projections_imgscale_' num2str(imgscale)],'conf');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%  KMSE_sub computing the regressors  %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CCR_PPs = [];
    %num_centers_use = 30;       % num_centers_use nearest centers to use 
    %nn_patch_max = 2048;        % max number of the nn patches to each cluster center
    fname = ['CCR_x' num2str(upscaling) '_K' num2str(dict_sizes(d)) '_max' num2str(nn_patch_max) '_d' num2str(num_centers_use) '_lambda' num2str(llambda_CCR*1000) '.mat'];
    %fname = ['kmeans_sub_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(nn_patch_max) 'nn_' num2str(num_centers_use) 'sub_whole.mat'];
    % e.g. kmeans_sub_x3_1024atoms_2048nn_40sub_whole.mat
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute CCR regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf_Kmeans, load_images(...            
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
        clear l2n
        l2n_h = repmat(l2,size(phires,1),1);
        clear l2
        phires = phires./l2n_h;
        clear l2n_h

        %llambda_kmeans_sub = 0.1;
        %cluster the whole data with kmeans plus plus
        folder_current = pwd;
        run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);
        [centers, assignments] = vl_kmeans(plores, dict_sizes(10), 'Initialization', 'plusplus');
        assignments = double(assignments);
        %load('cluster_x3_whole.mat');
        for i = 1:size(conf.dict_lores, 2)
            fprintf('Compute CCR regressors: %d/%d\n', i, size(conf.dict_lores,2));
            D = pdist2(single(centers'), single(conf_Kmeans.dict_lore_kmeans(:, i)'));
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
                sub_D = pdist2(single(sub_plores'), single(conf_Kmeans.dict_lore_kmeans(:, i)'));
                [~, sub_idx] = sort(sub_D, 'ascend');
                Lo = sub_plores(:, sub_idx(1:nn_patch_max));
                Hi = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                Lo = sub_plores;
                Hi = sub_phires;
            end
              
           
           CCR_PPs{i} = Hi*((Lo'*Lo+llambda_CCR*eye(size(Lo,2)))\Lo');
        end
        
     
        clear plores
        clear phires
        clear sub_plores
        clear sub_phires
        
        ttime = toc(ttime);        
        save(fname,'CCR_PPs','ttime', 'number_samples');   
        toc
    end 
    

    %%    
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);
    conf.result_dirRGB = qmkdir(['ResRGB-' sprintf('%s_x%d-', input_dir, upscaling) 'k' num2str(dict_sizes(d)) '_d' num2str(num_centers_use) '_max' num2str(nn_patch_max) '_lam' num2str(llambda_CCR*1000)]);
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
        % A+
        if ~isempty(Aplus_PPs)
            fprintf('A+\n');
            conf.PPs = Aplus_PPs;
            startt = tic;
            res{9} = scaleup_ANR(conf, low);
            toc(startt)
            conf.countedtime(9,i) = toc(startt);    
        else
            res{9} = interpolated;
        end        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%  SR via CCR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~isempty(CCR_PPs)
            fprintf('CCR\n');
            conf_Kmeans.PPs = CCR_PPs;
            startt = tic;
            res{10} = scaleup_CCR(conf_Kmeans, low);
            toc(startt)
            conf.countedtime(10,i) = toc(startt);
            
        else
            res{10} = interpolated;
        end
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1}, res{8}{1}, ...
            res{9}{1}, res{10}{1});
        
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)            
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]); 
            imwrite(result(:, :, j), conf.results{i}{j});

            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = result(:,:,j);
            end
            
            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end        
        conf.filenames{i} = f;
    end   
    conf.duration = cputime - t;

    % Test performance
    % PSNR
    run_comparisonRGB_PSNR(conf); 


end
end
end
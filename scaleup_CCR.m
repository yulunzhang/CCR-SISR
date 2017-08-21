function [imgs, midres] = scaleup_CCR(conf, imgs)

% Super-Resolution Iteration
    fprintf('Scale-Up CCR');
    midres = resize(imgs, conf.upsample_factor, conf.interpolate_kernel);    
    
    for i = 1:numel(midres)
        features = collect(conf, {midres{i}}, conf.upsample_factor, conf.filters);
        features = double(features);

        % Reconstruct using patches' dictionary and their anchored
        % projections
                
        features = conf.V_pca'*features;
        
        patches = zeros(size(conf.PPs{1},1),size(features,2));

     
        D = pdist2(single(conf.dict_lore_kmeans)', single(features)');
        
        [val idx] = min(D);
        for l = 1:size(features,2)            
            patches(:,l) = conf.PPs{idx(l)} * features(:,l);
        end


        
        % Add low frequencies to each reconstructed patch        
        patches = patches + collect(conf, {midres{i}}, conf.scale, {});
        
        % Combine all patches into one image
        img_size = size(imgs{i}) * conf.scale;
        grid = sampling_grid(img_size, ...
            conf.window, conf.overlap, conf.border, conf.scale);
        result = overlap_add(patches, img_size, grid);
        imgs{i} = result; % for the next iteration
        fprintf('.');
    end
fprintf('\n');

% This MATLAB file helps to read features using a caffe model

clear all; close all; clc;

addpath ./matlab

%
modelName = 'my_deploy_file.prototxt';
modelWeight = 'path_to_caffemodel/my_model.caffemodel';

sidePf = {'l', 'r'};

faceDBDir = '/my_hdd_path/dbs/lfw_orig/';
faceImgDir = 'N_112_96_mtcnn/';

% caffe.set_mode_cpu();
caffe.set_device(1);
caffe.set_mode_gpu();

% Define net
net = caffe.Net(modelName, modelWeight, 'test');
batSize = size(net.blobs('data').get_data(), 4);

% Database related information
db_dir = strcat(faceDBDir, faceImgDir);
feature_length = 512;
inpImHt = 112;
inpImWd = 96;

clc;
clear saveFileName
for si = 1:2
    saveFileName{si} = char(strcat('/my_hdd_path/caffe/tmp_eval_', sidePf(si), '.mat'));
    
    %% Processing left pair
    display('## Processing left pair')
    fid = fopen(char(strcat('/my_hdd_path/casiaex/file_lists/lfwtest_p1_norm_', sidePf(si), '.txt')));
    imageLabelInfo = textscan(fid, '%s %d');
    imageLabelList = imageLabelInfo{1};
    
    num_images = length(imageLabelList);
    
    % Redefine number of images w.r.t. batch size
    num_images = (num_images/batSize)*batSize;
    
    data_left = zeros(num_images,feature_length,'single');
    
    bI = 1; % begin Index
    while (bI<num_images)
        %bI
        if (mod(bI, 3000) == 1)
            display(strcat(': Processed ', 32, num2str(bI), 32, ' files'));
        end
        
        stIndx = bI;
        edIndx = bI+batSize-1;
        
        cbindx = 1;
        clear J
        for i = stIndx:edIndx
            % cbindx
            imfile_left = imageLabelList{i}; % label = 0 or 1
            imfile_left = strcat(imfile_left(1:end-3), 'jpg');
            
            %%%%%% Feature processing part
            leftImgName = strcat(db_dir, imfile_left);
            
            if(si==1)
                tpos = strfind(leftImgName, '-l');
            else
                tpos = strfind(leftImgName, '-r');
            end
            
            leftImgName = char(strcat(leftImgName(1:tpos(end)-1), '.jpg'));
            
            cropImg = imread(leftImgName);
                
                if size(cropImg, 3) > 1
                    cropImg = rgb2gray(cropImg);
                end
                
                cropImg = single(cropImg);
                cropImg = (cropImg - 127.5)/128;
                
                if(si==2)
                    cropImg = permute(flipdim(cropImg, 2), [2 1 3]);
                else
                    cropImg = permute(cropImg, [2 1 3]);
                end
                
                J(:,:,1,cbindx) = cropImg;
            
            cbindx = cbindx+1;
        end
        
        % Run Network
        net.forward({J});
        fts = net.blobs('fc5').get_data();
        fts = permute(fts, [4 3 2 1]);
        data_left(stIndx:edIndx, :) = fts;
        
        % Change index to the next batch
        bI = bI+batSize;
    end
    
    %% Processing right pair
    display('## Processing right pair')
    fid = fopen(char(strcat('/my_hdd_path/casiaex/file_lists/lfwtest_p2_norm_', sidePf(si) ,'.txt')));
    imageLabelInfo = textscan(fid, '%s %d');
    imageLabelList = imageLabelInfo{1};
    
    num_images = length(imageLabelList);
    
    % Redefine number of images w.r.t. batch size
    num_images = (num_images/batSize)*batSize;
    
    data_right = zeros(num_images,feature_length,'single');
    
    bI = 1; % begin Index
    while (bI<num_images)
        
        if (mod(bI, 3000) == 1)
            display(strcat(': Processed ', 32, num2str(bI), 32, ' files'));
        end
        
        stIndx = bI;
        edIndx = bI+batSize-1;
        
        cbindx = 1;
        clear J
        for i = stIndx:edIndx
            % i
            imfile_right = imageLabelList{i}; % label = 0 or 1
            imfile_right = strcat(imfile_right(1:end-3), 'jpg');
            
            %%%%%% Feature processing part
            rightImgName = strcat(db_dir, imfile_right);
            
            if(si==1)
                tpos = strfind(rightImgName, '-l');
            else
                tpos = strfind(rightImgName, '-r');
            end
            
            rightImgName = char(strcat(rightImgName(1:tpos(end)-1), '.jpg'));
            
            cropImg = imread(rightImgName);
                
                if size(cropImg, 3) > 1
                    cropImg = rgb2gray(cropImg);
                end
                
                cropImg = single(cropImg);
                cropImg = (cropImg - 127.5)/128;
                
                if(si==2)
                    cropImg = permute(flipdim(cropImg, 2), [2 1 3]);
                else
                    cropImg = permute(cropImg, [2 1 3]);
                end
                
                J(:,:,1,cbindx) = cropImg;
            
            cbindx = cbindx+1;
        end
        
        % Run Network
        net.forward({J});
        fts = net.blobs('fc5').get_data();
        fts = permute(fts, [4 3 2 1]);
        data_right(stIndx:edIndx, :) = fts;
        
        % Change index to the next batch
        bI = bI+batSize;
    end
    
    %%%%%% Create HDF5 database
    display('Saving features ...');
    save(char(saveFileName{si}), 'data_left', 'data_right');    
end

% If I need get evaluation results, e.g., accuracy, TAR@FAR etc..
% [allAcc allTaf] = lfw_roc_analysis(saveFileName);

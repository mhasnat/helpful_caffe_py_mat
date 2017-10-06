# MATLAB function to obtain manual facial landmarks

clear all; clc; close all;
warning off;

lpts = [37 40 43 46 18 20 22 23 25 27 33 35 49 52 55];

% existing missing file
fid = fopen('/home/hasnat/Desktop/Face/DBs/CACD/new_missing.txt');
imageLabelInfo = textscan(fid, '%s %d');
imageLabelList = unique(imageLabelInfo{1});

% Which image
num_images = length(imageLabelList);

trgFile_dir = '/home/hasnat/Desktop/Face/DBs/CACD/LM_CACD_VS/';

for i=1:num_images
    crImg = imageLabelList{i}; % label = 0 or 1
    
    figure;
    imshow(crImg);
    fig = gcf;
    [x, y] = getpts(fig);
    
    
    shapeXY = zeros(68,2);
    shapeXY(lpts, 1) = x(1:15);
    shapeXY(lpts, 2) = y(1:15);
    
    pp = strfind(crImg, '/');
    saveMatfile = crImg(pp(end)+1:length(crImg)-4);
    trgFile = strcat(trgFile_dir, saveMatfile, '.mat');
    save(trgFile, 'shapeXY');
    
    i
    close all
    % pause;
end

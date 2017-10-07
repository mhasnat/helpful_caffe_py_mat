clc; clear all; % close all;
warning off;

%% Setup Caffe
addpath('PATH_TO/caffe/matlab');
caffe.reset_all();

% load face model and creat network
desPath = 'PATH_TO/caffe/';
caffe.set_device(0);
caffe.set_mode_gpu();
model = [desPath 'face_deploy.prototxt'];
weights = 'PATH_TO/face_model.caffemodel';

net = caffe.Net(model, weights, 'test');

%% Read file lists
rootDir = 'PATH_TO_Normalized_Face_Directory/';

% list of files ...
fid = fopen('PATH_TO/list.txt');
imageLabelInfo = textscan(fid, '%s %d');
imageList = imageLabelInfo{1};
labelList = imageLabelInfo{2};

num_images = length(imageList);

ftVecLen = 512;
ftVec = zeros(num_images, ftVecLen);
smImg = zeros(20,20,num_images);

for imNum=1:num_images
    imNum
    imgName = imageList{imNum};
    
    tImfileName = char(strcat(rootDir, imgName));
    
    ImageSample = imread(tImfileName);
    
    if size(ImageSample, 3) > 1
        ImageSample = rgb2gray(ImageSample);
    end
    
    smImg(:,:,imNum) = imresize(ImageSample, [20 20]);
    
    ImageSample = single(ImageSample);
    ImageSample = (ImageSample - 127.5)/128;
    ImageSample_l = permute(ImageSample, [2,1,3]);        
    ImageSample_r = permute(flipdim(ImageSample, 2), [2 1 3]);
    
    % extract deep feature
    net.forward({ImageSample_l});
    res = net.blobs('fc5').get_data();
    res = permute(res, [4 3 2 1]);
    
    net.forward({ImageSample_r});
    res_ = net.blobs('fc5').get_data();
    res_ = permute(res_, [4 3 2 1]);
    
    ftVec(imNum,:) = max(res, res_);
end

save(strcat('ftVec.mat'), 'ftVec', 'labelList', 'smImg');

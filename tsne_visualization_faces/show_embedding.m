% Show the tSNE map and place information/data-points (small images) in a 2D image
% Need to download code from github repo bhtsne-master: https://github.com/lvdmaaten/bhtsne
clc; clear all; % close all;
warning off;
addpath(genpath('PATH_TO/bhtsne-master'));

%% load embedding, features (X), labels (L), images (smI, we keep small size of the image) and other information
load(strcat(dbPath, '/', 'fmap_ftVec_labels_imgs.mat'));

idS = 0;
tIndices = find(labelList == idS);
tMap = map(tIndices, :);

N = length(tIndices);
smI = smImg(:,:,tIndices);

% 
x = bsxfun(@minus, tMap, min(tMap));
x = bsxfun(@rdivide, x, max(x));

%% create an embedding image
S = 10000; % size of full embedding image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every single image

Ntake = N;
for i=1:Ntake
    i
    if mod(i, 100)==0
        fprintf('%d/%d...\n', i, Ntake);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
    
    I = smI(:,:,i);
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end

imshow(G);

% %% do a guaranteed quade grid layout by taking nearest neighbor
% 
% S = 500; % size of final image
% G = zeros(S, S, 3, 'uint8');
% s = 50; % size of every image thumbnail
% 
% xnum = S/s;
% ynum = S/s;
% used = false(N, 1);
% 
% qq=length(1:s:S);
% abes = zeros(qq*2,2);
% i=1;
% for a=1:s:S
%     for b=1:s:S
%         abes(i,:) = [a,b];
%         i=i+1;
%     end
% end
% 
% for i=1:size(abes,1)
%     a = abes(i,1);
%     b = abes(i,2);
%     xf = (a-1)/S;
%     yf = (b-1)/S;
%     dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
%     dd(used) = inf; % dont pick these
%     [dv,di] = min(dd); % find nearest image
% 
%     used(di) = true; % mark as done
%     I = smI(:,:,di);
%     if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);
% 
%     G(a:a+s-1, b:b+s-1, :) = I;
% 
%     if mod(i,100)==0
%         fprintf('%d/%d\n', i, size(abes,1));
%     end
% end
% 
% imshow(G);

clc; clear all; % close all;
warning off;
addpath(genpath('PATH_TO/bhtsne-master'));

load('ftVec.mat') % features vectors and labels

X = ftVec;
L = labelList;

numDims = 2; pcaDims = 50; perplexity = 50; theta = .5; alg = 'svd';
map = fast_tsne(X, numDims, pcaDims, perplexity, theta, alg);
gscatter(map(:,1), map(:,2), L);

save('tsne_map.mat', 'map')

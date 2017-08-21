clear all;
clc;
close all;
run('H:\Projects\Clustering\vlfeat-0.9.19-bin\vlfeat-0.9.19\toolbox/vl_setup.m');%Vl_feat library for k-nearest neighbor queries
numData = 5e3 ;
dimension = 2 ;
data = rand(dimension,numData) ;
numClusters = 30 ;
tic
%[centers, assignments] = vl_kmeans(data, numClusters, 'Initialization', 'plusplus','MaxNumIterations',100000) ;
[centers, assignments] = vl_kmeans(data, numClusters, 'Initialization', 'plusplus') ;
toc
% features= [1 2 3 4 5;
%            6 7 8 9 10;
%            11 12 13 14 15];
% 
% num_nn=4;
% tic
% [ind distance]=vl_kdtreequery(vl_kdtreebuild(features),features,features,'numneighbors',num_nn);
% toc

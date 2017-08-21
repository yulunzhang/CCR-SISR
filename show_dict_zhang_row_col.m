%% show the dictionary

function [h] = show_dict_zhang_row_col()
clear all; close all; clc
loaddata = load('conf_Kmeans_pp_1024_finalx3.mat');
A = loaddata.conf_Kmeans_pp.dict_lore_kmeans;
figure;
warning off all

if exist('figstart', 'var') && ~isempty(figstart), figure(figstart); end

[L M]=size(A);
if ~exist('numcols', 'var')
    numcols = ceil(sqrt(L));
    while mod(L, numcols), numcols= numcols+1; end
end
ysz = numcols;
xsz = ceil(L/ysz);

% m=floor(sqrt(M*ysz/xsz));
% n=ceil(M/m);
m = 32;
n = 32;
colormap(gray)

buf=1;
array=ones(buf+m*(xsz+buf),buf+n*(ysz+buf));

k=1;
for i=1:m
    for j=1:n
        if k>M continue; end
        clim=max(abs(A(:,k)));
        array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz])=reshape(A(:,k),xsz,ysz)/clim;
        k=k+1;
    end
end

if isreal(array)
    h=imagesc(array,'EraseMode','none',[-1 1]);
else
    h=imagesc(20*log10(abs(array)),'EraseMode','none',[-1 1]);
end;
axis image off

drawnow

warning on all
%print -dpng -r600 clusters_Euclidean_x3_1024atoms_32x32.png
end
load('TrainingSamplesDCT_subsets_8.mat');

TrainsampleDCT_FG = D1_FG;
TrainsampleDCT_BG = D1_BG;
N_FG = size(TrainsampleDCT_FG,1);
N_BG = size(TrainsampleDCT_BG,1);

%% Part A
%Prior calculations: 
total = N_FG+N_BG;
P_FG = N_FG/total;
P_BG = N_BG/total;
load('data.mat')

%% Part B

%full plots
mu_cheetah = mean(TrainsampleDCT_FG);
cov_cheetah = cov(TrainsampleDCT_FG); % 64x64 matrix

% Calculate mean and covariance for the grass class
mu_grass = mean(TrainsampleDCT_BG); % 1x64 vector
cov_grass = cov(TrainsampleDCT_BG); % 64x64 matrix

%% Classification part for 64 Gaussians 
img = imread('cheetah.bmp');
[numRows, numCols] = size(img);
Img_decision = zeros(numRows, numCols);
cnt = 1; 
for i = 1:numRows 
    for j = 1:numCols
        logPxgivenCheeta = 0;
        logPxgivenGrass = 0;
        for s = 1:64
            x = Data_array(cnt, s);
            logPxgivenCheeta = logPxgivenCheeta + likliehood_FG(mu_cheetah, cov_cheetah, s, x);
            logPxgivenGrass = logPxgivenGrass + likliehood_BG(mu_grass, cov_grass, s, x);
        end
        PCheetagivenX = exp(logPxgivenCheeta + log(P_FG));
        PGrassgivenX = exp(logPxgivenGrass + log(P_BG));
        
        Img_decision(i,j) = PCheetagivenX >= PGrassgivenX;
        cnt = cnt + 1;
    end
end


%% create mask 

A =Img_decision;
figure;
imagesc(A);
colormap(gray(225));  
           
title('Decision Mask for Cheetah');

%% ERROR

cheetah_mask = imread('cheetah_mask.bmp');  
cheetah_mask = im2bw(cheetah_mask);         

[numRows, numCols] = size(cheetah_mask);
errors = 0;
for i = 1:numRows
    for j = 1:numCols
        if A(i,j) ~= cheetah_mask(i,j)
            errors = errors +1;
        end 
    end
end

Percenterror = errors/(numRows*numCols)
figure;
imagesc(cheetah_mask);
colormap(gray(225)); 

%% Part C function
function Data_array = data_fill(blockSize, zigZag,A1,block_num, Data_array)
    
    for i = 1:blockSize(2)
        for j = 1:blockSize(1)
            idx = zigZag(i, j) + 1; 
            Data_array(block_num, idx) = A1(i, j);  
        end
    end
end

function logPxgivenCheeta = likliehood_FG(mu_cheetah, cov_cheetah, k, x) 
    mu_c = mu_cheetah(k);
    sigma_c = sqrt(cov_cheetah(k, k));
    logPxgivenCheeta = -0.5 * log(2 * pi) - log(sigma_c) - (0.5 * ((x - mu_c) ./ sigma_c).^2);
end

function logPxgivenGrass = likliehood_BG(mu_grass, cov_grass, k, x) 
    mu_c = mu_grass(k);
    sigma_c = sqrt(cov_grass(k, k));
    logPxgivenGrass = -0.5 * log(2 * pi) - log(sigma_c) - (0.5 * ((x - mu_c) ./ sigma_c).^2);
end


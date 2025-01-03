load('TrainingSamplesDCT_subsets_8.mat')
load('Prior_1.mat')
load("Alpha.mat")
img = imread('cheetah.bmp');






%% Part a

% Preallocate storage for Sigma_0 matrices
num_alphas = length(alpha);  % Number of alpha values
%Sigma_0_all = cell(num_alphas, 1);  % Store Sigma_0 for each alpha
alpha_idx = 9;
alpha(alpha_idx)
Sigma_0 = alpha(alpha_idx) * diag(W0);


Train_FG = D4_FG;
Train_BG = D4_BG;

FG_sample_mean = mean(Train_FG,1);
FG_sample_cov = cov(Train_FG,1);
BG_sample_mean = mean(Train_BG,1);
BG_sample_cov = cov(Train_BG,1);

N = size(Train_FG,1);
FG_Sigma_0 = alpha(alpha_idx) * diag(W0);

x = inv((FG_Sigma_0+ (1/N)*FG_sample_cov));
FG_posterior_mean = FG_Sigma_0* x * FG_sample_mean'+ (1/N)*FG_Sigma_0* x *mu0_FG';

FG_posterior_sigma = FG_Sigma_0 * x * FG_sample_cov /N + FG_sample_cov;
FG_Bayes_covinv = inv(FG_posterior_sigma);
log_FG_det = sum(log(eig(FG_Bayes_covinv)));




N = size(Train_BG,1);
BG_Sigma_0 = alpha(alpha_idx) * diag(W0);

x = inv((BG_Sigma_0+ (1/N)*BG_sample_cov));
BG_posterior_mean = BG_Sigma_0* x * BG_sample_mean'+ (1/N)*BG_Sigma_0* x *mu0_BG';

BG_posterior_sigma = BG_Sigma_0 * x * BG_sample_cov /N + BG_sample_cov;
BG_Bayes_covinv = inv(BG_posterior_sigma);
log_BG_det = sum(log(eig(BG_Bayes_covinv)));

N_FG = size(Train_FG,1);
N_BG = size(Train_BG,1);
total = N_FG + N_BG;
P_FG = N_FG/total;
P_BG = N_BG/total; 
N = 64;
umap_FG = inv(inv(FG_Sigma_0) + N*inv(FG_sample_cov))*(inv(FG_Sigma_0)*mu0_FG'+N*inv(FG_sample_cov)*FG_sample_mean');
umap_BG = inv(inv(BG_Sigma_0) + N*inv(BG_sample_cov))*(inv(BG_Sigma_0)*mu0_BG'+N*inv(BG_sample_cov)*BG_sample_mean');


%% Classification part for 64 Gaussians 
load("data.mat")
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
            logPxgivenCheeta = logPxgivenCheeta + likliehood_FG(umap_FG, FG_posterior_sigma, s, x);
            logPxgivenGrass = logPxgivenGrass + likliehood_BG(BG_posterior_mean, BG_posterior_sigma, s, x);
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

Percenterror = (errors/(numRows*numCols))*100
%figure;
%imagesc(cheetah_mask);
%colormap(gray(225)); 

%% functions
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

%%







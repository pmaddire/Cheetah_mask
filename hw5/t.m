% Load Data
load('TrainingSamplesDCT_8_new.mat'); % Load training samples
load('data.mat'); 
% Parameters
Train_FG = TrainsampleDCT_FG;
Train_BG = TrainsampleDCT_BG;

%n = 64;  % Size of the matrix (5x5)
%sigma = rand(1, n);  % Generate a 1x5 vector of random numbers
%sigma_guess = diag(sigma);  % Create a diagonal matrix using the vector

n = 3;  % Dimensionality of x (adjust as needed)
K = 8;  % Number of components

pi_guess = 1/K * ones(1, K);  % Uniform prior guess for mixture weights
num_matrices = K;
% Initialize the mean vectors for each component (size: K x n)
mean_guess = rand(K, n);  % K components, each with n-dimensional mean vector
% Generate 8 random sigmas and create diagonal matrices
for i = 1:num_matrices
    sigma = rand(1, n)*1.5;  % Random sigma value between 0 and 1
    sigma_matrices{i} =  diag(sigma);  % 64x64 diagonal matrix with sigma on the diagonal
end
num_FG = length(Train_FG);
num_BG = length(Train_BG);

H_FG = zeros(num_FG, K); 
H_BG = zeros(num_BG, K); 

Dat_FG = Train_FG(:,1:n);
Dat_BG = Train_BG(:,1:n);

cheetah_mask = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');

%mean_guess(1,:)
tolerance = 0.05
%H = computeResponsibilityMatrix(Dat_FG, pi_guess, mean_guess, sigma_matrices);
%[pi_new, mean_new, sigma_new] = maximization_step(Dat_FG,H,K);
[u_FG, pi_FG, sigma_FG] = EM_algorithm(Dat_FG, mean_guess, pi_guess, sigma_matrices, tolerance);
[u_BG, pi_BG, sigma_BG] = EM_algorithm(Dat_BG, mean_guess, pi_guess, sigma_matrices, tolerance);
decision_mat = classifier(Data_array(:,1:n),img,u_FG,pi_FG,sigma_FG,u_BG,pi_BG,sigma_BG,num_BG,num_FG);
A =decision_mat;
error = errorpix(cheetah_mask, A)
figure;
imagesc(A);
colormap(gray(225));  
           
title('Decision Mask for Cheetah');


%%


%% Load Data
load('TrainingSamplesDCT_8_new.mat'); % Load training samples
load('data.mat');

num_mix = 5;
C = [1 2 4 8 16 32];
dim = [1 2 4 6 8 16 24 32 40 48 56 64];

tolerance = 0.01;
max_atemp = 200;

Train_FG = TrainsampleDCT_FG;
Train_BG = TrainsampleDCT_BG;

test = Train_FG(:,1:5);

N_FG = length(Train_FG);
N_BG = length(Train_BG);

P_FG = N_FG/(N_BG+N_FG);
P_BG = N_BG/(N_BG+N_FG);

cheetah_mask = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');

[numRows, numCols] = size(img);
A = zeros(numRows, numCols);

FG_mix = cell(1,num_mix);
BG_mix = cell(1,num_mix);

for i= C
    [FG_mixprob, FG_mean, FG_sigma] = learn_mix(Train_FG, max_atemp, i, tolerance);
    [BG_mixprob, BG_mean, BG_sigma] = learn_mix(Train_BG, max_atemp, i, tolerance);

    FG_mix{1} = {FG_mean, FG_sigma, FG_mixprob};
    BG_mix{1} = {BG_mean, BG_sigma, BG_mixprob};

    for d = dim
        FG_predict = likelihoods(FG_mean(:,1:D), FG_sigma(1:D, :), FG_mixprob,Data_array, P_FG);
        BG_predict = likelihoods(BG_mean(:,1:D), BG_sigma(1:D, :), BG_mixprob,Data_array, B_FG);
        predict = (FG_predict > BG_predict);
        error_list = (predict ~= cheetah_mask);
        p_errors( i= C, d=dim) = mean(error_list, 'all');
    end 
end 

%%

load TrainingSamplesDCT_8_new.mat
load freqs.mat

Data_array = freqs;
x = Data_array; 

C = [1 2 4 8 16 32];



dim = [1 2 4 8 16 24 32 40 48 56 64];

tol = 0.01;
maxiter = 1000;

perrors = zeros(length(C), length(dim));

pycheetah = size(TrainsampleDCT_FG, 1);
pygrass = size(TrainsampleDCT_BG, 1);

pygrass = pygrass / (pycheetah + pygrass);
pycheetah = 1 - pygrass;

A = zeros(original_size);
cheetah_mixtures = cell(1, 5);
grass_mixtures = cell(1, 5);

for c = C
    [cheetah_mu, cheetah_Sigma, cheetah_mixture_probabilities] = learn_mix(TrainsampleDCT_FG, maxiter, c, tol);
    [grass_mu, grass_Sigma, grass_mixture_probabilities] = learn_mix(TrainsampleDCT_BG, maxiter, c, tol);
    
    for D = dim
        cheetah_predictions = likelihoods(pycheetah, cheetah_mu(:, 1:D), cheetah_Sigma(1:D, :), cheetah_mixture_probabilities, freqs);
        grass_predictions = likelihoods(pygrass, grass_mu(:, 1:D), grass_Sigma(1:D, :), grass_mixture_probabilities, freqs);
        
        predictions = (cheetah_predictions > grass_predictions);
        error_mat = (predictions ~= cheetah_mask);
        
        p_errors(c == C, D == dim) = mean(error_mat, 'all');
    end
end

save Qb.mat p_errors

%%
load Qb.mat

dim = [1 2 4 8 16 24 32 40 48 56 64];
C = [1 2 4 8 16 32];

figure()
hold on

for i = 1:length(C)
    plot(dim, p_errors(i, :))
end

hold off

legend('C = 1', 'C = 2', 'C = 4', 'C = 8', 'C = 16', 'C = 32')

%%

load TrainingSamplesDCT8new.mat
load freqs.mat
C = [1 2 4 8 16 32];

dim = [1 2 4 8 16 24 32 40 48 56 64];
tol = 0.01;
max_iter = 1000;
errors = zeros(length(C), length(dim));

py_cheetah = size(TrainsampleDCT_FG, 1);
py_grass = size(TrainsampleDCT_BG, 1);

py_grass = py_grass / (py_cheetah + py_grass);
py_cheetah = 1 - py_grass;

A = zeros(original_size);
cheetah_mixtures = cell(1, 5);
grass_mixtures = cell(1, 5);

for c = C
    [cheetah_mu, cheetah_Sigma, cheetah_mixture_probabilities] = learn_mixture(TrainsampleDCT_FG, c, max_iter, tol);
    [grass_mu, grass_Sigma, grass_mixture_probabilities] = learn_mixture(TrainsampleDCT_BG, c, max_iter, tol);
    
    for D = dim
        cheetah_predictions = compute_likelihoods(py_cheetah, cheetah_mu(:, 1:D), cheetah_Sigma(1:D, :), cheetah_mixture_probabilities, freqs);
        grass_predictions = compute_likelihoods(py_grass, grass_mu(:, 1:D), grass_Sigma(1:D, :), grass_mixture_probabilities, freqs);
        
        predictions = (cheetah_predictions > grass_predictions);
        error_mat = (predictions ~= cheetah_mask);
        errors(c == C, D == dim) = mean(error_mat, 'all');
    end
end

% save quiz5b.mat errors
save quiz5b.mat errors


%%
load('Q5.mat')
perrors = p_errors;

dim = [1 2 4 8 16 24 32 40 48 56 64];

for i = 1:5
    figure()
    plot(dim, squeeze(perrors(i, 1, :)), 'Color', 'c')    
    hold on
    plot(dim, squeeze(perrors(i, 2, :)), 'Color', 'm')    
    plot(dim, squeeze(perrors(i, 3, :)), 'Color', 'b')    
    plot(dim, squeeze(perrors(i, 4, :)), 'Color', 'r')    
    plot(dim, squeeze(perrors(i, 5, :)), 'Color', 'g')    
        hold off
    legend('FG Mixture 1', 'FG Mixture 2', 'FG Mixture 3', 'FG Mixture 4', 'FG Mixture 5')
   % titlestr = sprintf('BG Mixture %d', i);
    title('BG Mixture %d', i);
    xlabel('Dimension')
    ylabel('Percent Error')
    filestr = sprintf('TeX/images/BG Mixture %d', i);
    saveas(gcf, filestr, 'epsc');
end


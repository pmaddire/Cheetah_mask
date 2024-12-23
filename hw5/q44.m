%%

load TrainingSamplesDCT_8_new.mat


C = [1 2 4 8 16 32];



dim = [1 2 4 8 16 24 32 40 48 56 64];

tolerance = 0.01;
max_atemp = 200;

Train_FG = TrainsampleDCT_FG;
Train_BG = TrainsampleDCT_BG;

N_FG = size(Train_FG, 1);  % Number of rows in Train_FG
N_BG = size(Train_BG, 1);  % Number of rows in Train_BG

P_FG = N_FG/(N_BG+N_FG);
P_BG = N_BG/(N_BG+N_FG);

cheetah_mask = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');

[numRows, numCols] = size(img);
A = zeros(numRows, numCols);

FG_mix = cell(1,num_mix);
BG_mix = cell(1,num_mix);
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


load('TrainingSamplesDCT_8_new.mat'); % Load training samples
load('data.mat');




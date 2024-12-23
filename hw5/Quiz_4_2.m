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



%% Functions 

function [u, piw, Sigma] = EM(x, u_g, pi_g, sigma_g)

u_old = u_g; 
pi_old = pi_g;
sigma_old = sigma_g; 

H = compute_H(x, pi_g, u_g, sigma_old);

end

function H = compute_H(x, piw, u, sigma)
    H = zeros(length(x), length(u))
    

end 


function hij = computeResponsibilityMatrix(X, pi_guess, mean_guess, sigma_matrices)
    % X: n x d matrix of data points (n data points, each of dimension d)
    % pi_guess: 1 x K vector of mixture component priors (K components)
    % mean_guess: K x d matrix of component means (K components, d-dimensional mean)
    % sigma_matrices: 1 x K cell array of diagonal covariance matrices (each d x d matrix)

    [n, d] = size(X);  % Number of data points (n) and dimensions (d)
    K = length(pi_guess);  % Number of components (K)
    
    % Initialize the responsibility matrix hij (n x K)
    hij = zeros(n, K);
    
    % Loop over each data point
    for i = 1:n
        % Get the data point x_i
        x_i = X(i, :)';
        
        % Compute the likelihood for each component j
        likelihoods = zeros(1, K);
        for j = 1:K
            mu_j = mean_guess(j, :)';  % Mean of component j (d x 1)
            sigma_j = sigma_matrices{j};  % Covariance of component j (d x d)
            
            % Multivariate Gaussian likelihood for data point x_i under component j
            % Compute (x_i - mu_j)' * sigma_j^-1 * (x_i - mu_j)
            diff = x_i - mu_j;
            likelihoods(j) = (1 / ((2*pi)^(d/2) * sqrt(det(sigma_j)))) * exp(-0.5 * diff' / sigma_j * diff);
        end
        
        % Compute the total likelihood (sum of all weighted likelihoods)
        total_likelihood = sum(pi_guess .* likelihoods);
        
        % Compute the responsibility h_ij for each component j
        hij(i, :) = (pi_guess .* likelihoods) / total_likelihood;
    end
end




function prob = Gaussian2(x, mu, Sigma)
    % x: Data point (column vector)
    % mu: Mean vector (column vector)
    % Sigma: Covariance matrix
    

    % Compute the Mahalanobis distance
    diff = x - mu;

    
    % Compute the probability using the Gaussian PDF formula
    prob = (1 / (sqrt(2 * pi) * sigma)) * exp(-0.5 * ((diff / sigma)^2));
end

function prob = Gaussian(x, mu, Sigma)
    % x: Data point (column vector)
    % mu: Mean vector (column vector)
    % Sigma: Covariance matrix
    
    % Compute the inverse and determinant of the covariance matrix
    
    Sigma_inv = inv(Sigma);
    Sigma_det = det(Sigma);
    
    % Compute the dimensionality
    d = length(mu);

    % Compute the Mahalanobis distance
    diff = x - mu;
    mahalanobisDist = diff' * Sigma_inv * diff;
    
    % Compute the probability using the Gaussian PDF formula
    prob = (1 / ((2 * pi)^(d / 2) * sqrt(Sigma_det))) * exp(-0.5 * mahalanobisDist);
end

function decision_mat = classifier(data,img,u_FG,pi_FG,sigma_FG,u_BG,pi_BG,sigma_BG,N_BG,N_FG)
    [numRows, numCols] = size(img);
    Img_decision = zeros(numRows, numCols);
    P_FG = N_FG/(N_BG+N_FG);
    P_BG = N_BG/(N_BG+N_FG);
    cnt = 1;
    for i = 1:numRows 
        for j = 1:numCols
            x = data(cnt,:);
            logPxgivenCheeta = likliehood(pi_FG, x, u_FG, sigma_FG);
            logPxgivenGrass =  likliehood(pi_BG, x, u_BG, sigma_BG);
            
            PCheetagivenX = exp(logPxgivenCheeta + log(P_FG));
            PGrassgivenX = exp(logPxgivenGrass + log(P_BG));
            
            Img_decision(i,j) = PCheetagivenX >= PGrassgivenX;
            cnt = cnt + 1;
        end
    end
    decision_mat = Img_decision;
end
function prob = likliehood(pi, x, u, sigma)
    prob =0;
    for j=1:length(u(:,1))
        %u(j,:)
        %x
        
        prob = prob + Gasussian(x', u(j,:)', sigma{j})*pi(j);
    end

end
function error = errorpix(cheetah_mask, A)
    [numRows, numCols] = size(cheetah_mask);
    errors = 0;
    for i = 1:numRows
        for j = 1:numCols
            if A(i,j) ~= cheetah_mask(i,j)
                errors = errors +1;
            end 
        end
    end
    
    error = errors/(numRows*numCols)
    %figure;
    %imagesc(cheetah_mask);
    %colormap(gray(225)); 

end



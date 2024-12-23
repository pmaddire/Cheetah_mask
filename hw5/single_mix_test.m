% Load Data
load('TrainingSamplesDCT_8_new.mat'); % Load training samples
load('data.mat'); 
% Parameters
Train_FG = TrainsampleDCT_FG;
Train_BG = TrainsampleDCT_BG;

%n = 64;  % Size of the matrix (5x5)
%sigma = rand(1, n);  % Generate a 1x5 vector of random numbers
%sigma_guess = diag(sigma);  % Create a diagonal matrix using the vector

n = 1;  % Dimensionality of x (adjust as needed)
K = 8;  % Number of components
pi_guess = 1/K * ones(1, K);  % Uniform prior guess for mixture weights

% Initialize the mean vectors for each component (size: K x n)
mean_guess = rand(K, n);  % K components, each with n-dimensional mean vector
% Generate 8 random sigmas and create diagonal matrices
for i = 1:num_matrices
    sigma = rand(1, n)*1.5;  % Random sigma value between 0 and 1
    sigma_matrices{i} =  diag(sigma);  % 64x64 diagonal matrix with sigma on the diagonal
end
num_FG = length(Train_FG);
num_BG = length(Train_BG);

%H_FG = zeros(num_FG, K); 
%H_BG = zeros(num_BG, K); 

Dat_FG = Train_FG(:,1:n);
Dat_BG = Train_BG(:,1:n);
%Dat_FG(1,:)'
%mean_guess(1,:)'
%sigma_matrices{1}

%Calculate H matrices (E Step)
%{
for j=1:K
    for i=1:num_FG
        H_FG(i,j) = Gasussian(Dat_FG(i,:)', mean_guess(j,:)', sigma_matrices{j});

    end

    for i=1:num_BG
        H_BG(i,j) = Gasussian(Dat_BG(i,:)', mean_guess(j,:)', sigma_matrices{j});
    end 

end 

prob = Gasussian(Dat_FG(1,:)', mean_guess(1,:)', sigma_matrices{1}) ;
%}

%H_FG = compute_H(Dat_FG, pi, mean_guess, sigma_matrices);
%H_BG = compute_H(Dat_BG, pi, mean_guess, sigma_matrices);

cheetah_mask = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');
[u_FG, pi_FG, sigma_FG] = EM(Dat_FG, mean_guess, pi_guess, sigma_matrices, K);
[u_BG, pi_BG, sigma_BG] = EM(Dat_BG, mean_guess, pi_guess, sigma_matrices, K);
decision_mat = classifier(Data_array(:,1:n),img,u_FG,pi_FG,sigma_FG,u_BG,pi_BG,sigma_BG,num_BG,num_FG);

A =decision_mat;
%figure;
%imagesc(A);
%colormap(gray(225));  
           
error = errorpix(cheetah_mask, A)
figure;
imagesc(A);
colormap(gray(225));  
           
title('Decision Mask for Cheetah');

%% test scripts

mean_guess(1,:)
length(Data_array(1,:))

%%

function [u, piw, Sigma] = EM(x, u_g, pi_g, sigma_g,K)
    %first iteration 
    %hold vals 
    
    u_old = u_g;
    pi_old = pi_g;
    sigma_old = sigma_g;

    %E step: 
    
    %H = compute_H(x, pi_g, u_g, sigma_g);
    H = computeResponsibilityMatrix(x, pi_old, u_old, sigma_old);
    
    %M step: 
     
    %[u_new, sigma_new, pi_new] = update(x, H, u_old);
    [pi_new, u_new, sigma_new] = maximization_step(x,H,K);
    %sigma_new{1}
    count =0;
    while ~convergence_check2(x, u_old, sigma_old, pi_old, u_new, sigma_new, pi_new )
        %x, old_u, old_sigma, old_pi, new_u, new_sigma, new_pi
        % hold old values before update: 
        %disp("in")
        u_old = u_new;
        pi_old = pi_new;
        sigma_old = sigma_new;
        %E step:
        %sigma_old
        %H = compute_H(x, pi_old, u_old, sigma_old);
        H = computeResponsibilityMatrix(x, pi_old, u_old, sigma_old);

        %M step: 
        %[u_new, sigma_new, pi_new] = update(x, H, u_old);
        [pi_new, u_new, sigma_new] = maximization_step(x,H,K);
        count = count+1
    end
    disp('exit')
    u = u_new;
    Sigma = sigma_new;
    piw = pi_new;
end


function H = compute_H(x, piw, u, Sigma)
   H = zeros(length(x), length(u)); 
   for j = 1: length(u(:,1))
    for i = 1:length(x)
        
        prob = Gasussian(x(i,:)', u(j,:)', Sigma{j}) * piw(j);
        
        reg = 0;
        for k = 1:length(u(:,1))

            reg = reg + Gasussian(x(i,:)', u(k,:)', Sigma{k}) * piw(k);
            
        end
        
        H(i,j) = prob/reg;
        
    end
   end
    
end

function prob = Gasussian(x, mu, Sigma)
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

function [u_new,sigma_new,pi_new] = update(x, H, u_old)
    u =u_old;
    u_new = u;
    for j = 1:length(u(:,1))
        n = length(x);
        w_sum = 0;
        reg =0;
        dist =0;
        for i=1:length(x)
            w_sum = w_sum + H(i,j)*x(i,:);
            dist = dist+ H(i,j) * (x(i,:) - u(j,:)).^2;
            reg = reg+ H(i,j);
        end
        sigma_new{j} = diag(dist/reg);
        epsilon = 1e-5; % curse of dimensionality 
        sigma_new{j} = sigma_new{j} + epsilon * eye(size(sigma_new{j}));
        u_new(j,:) = w_sum/reg;
        pi_new(j) = (1/n) * reg; 
    end
   
end

function conv = convergence_check(x, old_u, old_sigma, old_pi, new_u, new_sigma, new_pi)
   old_likelihood = 0;
   new_likelihood=0;
   for j = 1:length(old_u(:,1))
       for i = 1:length(x)
        %new_sigma{j}
        old_likelihood = old_likelihood + log(Gasussian(x(i,:)', old_u(j,:)', old_sigma{j}));
        new_likelihood = new_likelihood + log(Gasussian(x(i,:)', new_u(j,:)', new_sigma{j}));
       end
   end
  % old_likelihood = Gasussian(x, old_u, old_sigma);
   %new_likelihood = Gasussian(x, new_u, new_sigma);
   old_likelihood
   new_likelihood
   if(new_likelihood - old_likelihood)/ old_likelihood <= 0.01
        conv = 1
   else 
        conv =0
   end 
end 

function conv2 = convergence_check2(x, old_u, old_sigma, old_pi, new_u, new_sigma, new_pi )
    if (new_u-old_u)/old_u <= 0.01
        conv2 = 1;
    else 
        conv2 = 0;
    end
 
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

function [pi_new, mean_new, sigma_new] = maximization_step(X, H, K)
    % X: n x d matrix of data points (n points, d dimensions)
    % H: n x K matrix of responsibilities (hij matrix)
    % K: number of components
    
    [n, d] = size(X);  % n = number of data points, d = dimensionality of data
    
    % Update mixture weights (pi_j)
    pi_new = sum(H, 1) / n;  % pi_j = (sum of hij for each component) / n
    
    % Update means (mu_j)
    mean_new = (H' * X) ./ sum(H, 1)';  % mu_j = (sum of hij * x_i) / sum(hij)
    
    % Update covariance matrices (sigma_j)
    sigma_new = cell(1, K);  % Cell array to store covariance matrices for each component
    for j = 1:K
        % Compute the weighted covariance for each component
        X_centered = X - mean_new(j, :);  % Center the data around the new mean
        sigma_new{j} = (X_centered' * (H(:, j) .* X_centered)) / sum(H(:, j));
    end
end





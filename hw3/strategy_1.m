 load Prior_1.mat
load TrainingSamplesDCT_subsets_8.mat
load Alpha.mat

% vary alpha_ind as necessary
alpha_ind = 2;
subset = 1:64;
num_features = length(subset);

train_cheetah = D4_FG(:, subset);
train_grass = D4_BG(:, subset);

ridge = 0;

N = size(train_cheetah, 1);
cheetah_Sigma_0 = alpha(alpha_ind) * diag(W0(subset));
cheetah_Sigma_0_inv = pinv(cheetah_Sigma_0);
cheetah_sample_mean = mean(train_cheetah, 1);
cheetah_sample_cov = cov(train_cheetah, 1);
cheetah_sample_cov_inv = pinv(cheetah_sample_cov);
cheetah_intermediate_inv = pinv(cheetah_Sigma_0 + (1 / N) * cheetah_sample_cov);
cheetah_Bayes_mean = cheetah_Sigma_0 * cheetah_intermediate_inv * cheetah_sample_mean' + ...
                            (1 / N) * cheetah_Sigma_0 * cheetah_intermediate_inv * mu0_FG(subset)';
cheetah_Bayes_cov = cheetah_Sigma_0 * cheetah_intermediate_inv * cheetah_sample_cov / N + cheetah_sample_cov;
cheetah_Bayes_cov_inv = pinv(cheetah_Bayes_cov);

log_cheetah_det = sum(log(eig(cheetah_Bayes_cov + ridge * eye(num_features))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(train_grass,1);
grass_Sigma_0 = alpha(alpha_ind) * diag(W0(subset));
grass_Sigma_0_inv = pinv(grass_Sigma_0);
grass_sample_mean = mean(train_grass, 1);
grass_sample_cov = cov(train_grass, 1);
grass_sample_cov_inv = pinv(grass_sample_cov);
grass_intermediate_inv = pinv(grass_Sigma_0 + (1 / N) * grass_sample_cov);
grass_Bayes_mean = grass_Sigma_0 * grass_intermediate_inv * grass_sample_mean' + ...
                            (1 / N) * grass_Sigma_0 * grass_intermediate_inv * mu0_BG(subset)';
grass_Bayes_cov = grass_Sigma_0 * grass_intermediate_inv * grass_sample_cov / N + grass_sample_cov;
grass_Bayes_cov_inv = pinv(grass_Bayes_cov);

log_grass_det = sum(log(eig(grass_Bayes_cov + ridge * eye(num_features))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p_y_cheetah = size(train_cheetah, 1);
p_y_grass = size(train_grass, 1);

p_y_grass = p_y_grass / (p_y_cheetah + p_y_grass);
p_y_cheetah = 1 - p_y_grass;

%load freqs.mat
load data.mat
img = imread('cheetah.bmp');
[numRows, numCols] = size(img);
original_size = [numRows, numCols];
A = zeros(numRows, numCols);
freqs = Data_array;
%A = zeros(original_size);

for i = 1:original_size(1)
    for j = 1:original_size(2)
        cheetah_likelihood = -1/2 * (squeeze(freqs(i,j,subset)) - cheetah_Bayes_mean)' * cheetah_Bayes_cov_inv * (squeeze(freqs(i,j,subset)) - cheetah_Bayes_mean) - log_cheetah_det / 2 + log(p_y_cheetah);
        grass_likelihood = -1/2 * (squeeze(freqs(i,j,subset)) - grass_Bayes_mean)' * grass_Bayes_cov_inv * (squeeze(freqs(i,j,subset)) - grass_Bayes_mean) - log_grass_det / 2 + log(p_y_grass);
        if(cheetah_likelihood > grass_likelihood)
            A(i,j) = 1;
        end
    end
end

figure()
imagesc(A)
colormap(gray(255))
title('Classification Using the Best 8 Features', 'Interpreter', 'latex')

cheetah_mask = im2double(imread("cheetah_mask.bmp"));

p_error = sum(sum(abs(A - cheetah_mask))) / (numel(cheetah_mask))
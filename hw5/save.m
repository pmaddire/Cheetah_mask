load TrainingSamplesDCT_8_new.mat
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

%%
%load quiz5b.mat

dim = [1 2 4 8 16 24 32 40 48 56 64];
C = [1 2 4 8 16 32];

figure()
hold on

for i = 1:length(C)
    plot(dim, errors(i, :))
end

hold off

legend('C = 1', 'C = 2', 'C = 4', 'C = 8', 'C = 16', 'C = 32')


%%
function [means, covariances, mixture_probabilities] = learn_mixture(training_data, C, max_iter, tol)
    num_iter = 0;

    [N, dim] = size(training_data);

    h = zeros(N, C);  % h(i, j) = P{Z|X}(ej | xi; psi)
    means = training_data(randi(N, [C, 1]), :);  % C by dim matrix, random unique elements of training data
    covariances = rand(dim, C);  % random from U[0, 2]
    mixture_probabilities = rand(1, C);
    mixture_probabilities = mixture_probabilities / sum(mixture_probabilities);

    iterate_change = tol + 1;

    while (iterate_change > tol && num_iter < max_iter)
        prev_h = h;
        prev_means = means;
        prev_covariances = covariances;

        h = E_step(training_data, means, covariances, mixture_probabilities);
        [means, covariances, mixture_probabilities] = M_step(training_data, h, means);

        iterate_change = sum(abs(prev_means - means), 'all') / sum(abs(means), 'all') + ...
                         sum(abs(prev_covariances - covariances), 'all') / sum(abs(covariances), 'all') + ...
                         sum(abs(prev_h - h), 'all') / sum(abs(h), 'all');

        num_iter = num_iter + 1;
    end

    disp('Number of iterations:');
    disp(num_iter);
end

function [likelihoods] = compute_likelihoods(prior_probability, means, covariances, mixture_probabilities, freqs)
    %UNTITLED3 Summary of this function goes here
    % Detailed explanation goes here

    likelihoods = zeros(size(freqs, [1, 2]));
    [C, dim] = size(means);

    for k = 1:size(freqs, 1)
        for l = 1:size(freqs, 2)
            prob = 0;
            for c = 1:C
                diff = (squeeze(freqs(k, l, 1:dim))' - means(c, :)) .^ 2;
                % prob = prob + mixture_probabilities(c) * exp(-1/2 * diff * (1 ./ covariances(:, c))) / sqrt(prod(covariances(:, c)));
                temp = log(mixture_probabilities(c)) - 0.5 * (diff * (1 ./ covariances(:, c))) - 0.5 * sum(log(covariances(:, c)));
                prob = prob + exp(temp);
            end
            likelihoods(k, l) = log(prob) + log(prior_probability);
        end
    end
end
function [means, covariances, mixture_probabilities] = M_step(data, h, means)
    C = size(h, 2);
    [~, dim] = size(data);

    mixture_probabilities = mean(h, 1);
    covariances = zeros(dim, C);

    for k = 1:C
        normalization = sum(h(:, k));
        means(k, :) = sum(h(:, k) .* data) / normalization;
        
        diff = (data - means(k, :)) .^ 2;
        covariances(:, k) = sum(diff .* h(:, k)) / normalization;
    end

    covariances = covariances + 1e-5; % Small regularization term
end

function [h] = E_step(data, means, covariances, mixture_probabilities)
    C = length(mixture_probabilities);
    [N, ~] = size(data);
    h = zeros(N, C);
    
    for k = 1:C
        diff = (data - means(k, :)) .^ 2;
        h(:, k) = log(mixture_probabilities(k)) - 0.5 * (diff * (1 ./ covariances(:, k))) - 0.5 * sum(log(covariances(:, k)));
        
        % If the likelihood is too large, display it
        if (any(h(:, k) >= 700))
            disp(exp(h(:, k)))
        end
    end
    
    % Normalize the likelihoods
    h = exp(h);
    h = h ./ sum(h, 2);
end



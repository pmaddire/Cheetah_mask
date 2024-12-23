load("Alpha.mat");
alpha_percent_errors = compute_errors_for_all_alpha(alpha);  % 'alpha' is your array of alpha values
disp(alpha_percent_errors);


function alpha_percent_errors = compute_errors_for_all_alpha(alpha_values)
    % Load data
    load('TrainingSamplesDCT_subsets_8.mat');
    load('Prior_2.mat');
    load("Alpha.mat");
    img = imread('cheetah.bmp');
    
    % Preallocate storage for results
    num_alphas = length(alpha_values);  % Number of alpha values
    alpha_percent_errors = table('Size', [num_alphas, 2], 'VariableTypes', {'double', 'double'}, ...
                                 'VariableNames', {'Alpha', 'PercentError'});

    % Loop over each alpha value
    for alpha_idx = 1:num_alphas
        alpha_val = alpha_values(alpha_idx);
        
        % Compute the Sigma_0 for this alpha value
        Sigma_0 = alpha_val * diag(W0);
        
        % Training Data for Foreground (FG) and Background (BG)
        Train_FG = D4_FG;
        Train_BG = D4_BG;
        
        % Compute sample statistics for FG and BG
        FG_sample_mean = mean(Train_FG, 1);
        FG_sample_cov = cov(Train_FG, 1);
        BG_sample_mean = mean(Train_BG, 1);
        BG_sample_cov = cov(Train_BG, 1);
        
        % Inverse of covariance matrix for FG
        N_FG = size(Train_FG, 1);
       
        x_FG = inv((Sigma_0 + (1/N_FG) * FG_sample_cov));
        FG_posterior_mean = Sigma_0 * x_FG * FG_sample_mean' + (1/N_FG) * FG_sample_cov * x_FG * mu0_FG';
        FG_posterior_sigma = Sigma_0 * x_FG * FG_sample_cov / N_FG + FG_sample_cov;
        
        % Inverse of covariance matrix for BG
        N_BG = size(Train_BG, 1);
       
        x_BG = inv((Sigma_0 + (1/N_BG) * BG_sample_cov));
        BG_posterior_mean = Sigma_0 * x_BG * BG_sample_mean' + (1/N_BG) * BG_sample_cov * x_BG * mu0_BG';
        BG_posterior_sigma = Sigma_0 * x_BG * BG_sample_cov / N_BG + BG_sample_cov;

        % Probability of each class (FG, BG)
        N_FG = size(Train_FG, 1);
        N_BG = size(Train_BG, 1);
        total = N_FG + N_BG;
        P_FG = N_FG / total;
        P_BG = N_BG / total;
        
        % Classification part for 64 Gaussians
        load("data.mat");
        img = imread('cheetah.bmp');
        [numRows, numCols] = size(img);
        Img_decision = zeros(numRows, numCols);
        Img_decision2 = zeros(numRows, numCols);
        cnt = 1;
        error_list = zeros(num_alphas,1);
        % Classify each pixel
        for i = 1:numRows
            for j = 1:numCols
                logPxgivenCheeta = 0;
                logPxgivenGrass = 0;
                logPxgivenCheeta2 = 0;
                logPxgivenGrass2 = 0;
                for s = 1:64
                    x = Data_array(cnt, s);
                    logPxgivenCheeta = logPxgivenCheeta + likliehood_FG(FG_posterior_mean, FG_posterior_sigma, s, x);
                    logPxgivenGrass = logPxgivenGrass + likliehood_BG(BG_posterior_mean, BG_posterior_sigma, s, x);
                    logPxgivenCheeta2 = logPxgivenCheeta2 + likliehood_FG(FG_posterior_mean, FG_sample_cov, s, x);
                    logPxgivenGrass2 = logPxgivenGrass2 + likliehood_BG(BG_posterior_mean, BG_sample_cov, s, x);

                end
                PCheetagivenX = exp(logPxgivenCheeta + log(P_FG));
                PGrassgivenX = exp(logPxgivenGrass + log(P_BG));
                PCheetagivenX2 = exp(logPxgivenCheeta2 + log(P_FG));
                PGrassgivenX2 = exp(logPxgivenGrass2 + log(P_BG));
                Img_decision(i,j) = PCheetagivenX >= PGrassgivenX;
                Img_decision2(i,j) = PCheetagivenX2 >= PGrassgivenX2;
                cnt = cnt + 1;
            end
        end

        % Compute Error Mask
        cheetah_mask = imread('cheetah_mask.bmp');
        cheetah_mask = im2bw(cheetah_mask);
        [numRows, numCols] = size(cheetah_mask);
        errors = 0;
        errors2 = 0;
        for i = 1:numRows
            for j = 1:numCols
                if Img_decision(i,j) ~= cheetah_mask(i,j)
                    errors = errors + 1;
                end
                if Img_decision2(i,j) ~= cheetah_mask(i,j)
                    errors2 = errors2 + 1;
                end
            end
        end
        
        % Calculate Percent Error
        PercentError = (errors / (numRows * numCols)) * 100;
        PercentError2 = (errors2 / (numRows * numCols)) * 100; 
        error_list(alpha_idx) = PercentError;
        % Store the result in the table
        alpha_percent_errors{alpha_idx, 'Alpha'} = alpha_val;
        alpha_percent_errors{alpha_idx, 'PercentError'} = PercentError;
        %alpha_percent_errors{alpha_idx, 'PercentError2'} = PercentError2;
    end
    
    % Plot the log graph of Percent Error vs Alpha
    figure;
    semilogx(alpha_percent_errors.Alpha, alpha_percent_errors.PercentError, '-o');

    xlabel('Alpha');
    ylabel('Percent Error (%)');
    title('Log-Scale Plot of Percent Error vs Alpha');
    grid on;
end

%% Functions
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

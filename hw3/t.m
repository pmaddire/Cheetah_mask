alpha_percent_errors = compute_errors_for_all_alpha_D4(alpha);  % 'alpha' is your array of alpha values
disp(alpha_percent_errors);


function alpha_percent_errors = compute_errors_for_all_alpha_D4(alpha_values)
    % Load data
    load('TrainingSamplesDCT_subsets_8.mat');
    load('Prior_1.mat');
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
        %FG_posterior_mean = Sigma_0 * x_FG * FG_sample_mean' + (1/N_FG) * Sigma_0 * x_FG * mu0_FG';
        %FG_posterior_sigma = Sigma_0 * x_FG * FG_sample_cov / N_FG + FG_sample_cov;
        
        % Inverse of covariance matrix for BG
        N_BG = size(Train_BG, 1);
        x_BG = inv((Sigma_0 + (1/N_BG) * BG_sample_cov));
        %BG_posterior_mean = Sigma_0 * x_BG * BG_sample_mean' + (1/N_BG) * Sigma_0 * x_BG * mu0_BG';
        %BG_posterior_sigma = Sigma_0 * x_BG * BG_sample_cov / N_BG + BG_sample_cov;

        % Probability of each class (FG, BG)
        N_FG = size(Train_FG, 1);
        N_BG = size(Train_BG, 1);
        total = N_FG + N_BG;
        P_FG = N_FG / total;
        P_BG = N_BG / total;
        
        % UMAP for FG and BG
        N = 64;
        %umap_FG = inv(inv(Sigma_0) + N * inv(FG_sample_cov)) * (inv(Sigma_0) * mu0_FG' + N * inv(FG_sample_cov) * FG_sample_mean');
        %umap_BG = inv(inv(Sigma_0) + N * inv(BG_sample_cov)) * (inv(Sigma_0) * mu0_BG' + N * inv(BG_sample_cov) * BG_sample_mean');
        
        umap_FG = inv(Sigma_0 + (1/N_FG)*FG_sample_cov)*(Sigma_0*mu0_FG' + (1/N_FG)*FG_sample_cov*FG_sample_mean');
        umap_BG = inv(Sigma_0 + (1/N_BG)*BG_sample_cov)*(Sigma_0*mu0_BG' + (1/N_BG)*BG_sample_cov*BG_sample_mean');
        
        Sigma_map_FG = Sigma_0 + (1/N_FG)*FG_sample_cov;
        Sigma_map_BG = Sigma_0 + (1/N_BG)*BG_sample_cov;

        % Classification part for 64 Gaussians
        load("data.mat");
        img = imread('cheetah.bmp');
        [numRows, numCols] = size(img);
        Img_decision = zeros(numRows, numCols);
        cnt = 1;

        % Classify each pixel
        for i = 1:numRows
            for j = 1:numCols
                logPxgivenCheeta = 0;
                logPxgivenGrass = 0;
                for s = 1:64
                    x = Data_array(cnt, s);
                    logPxgivenCheeta = logPxgivenCheeta + likliehood_FG(umap_FG, Sigma_map_FG, s, x);
                    logPxgivenGrass = logPxgivenGrass + likliehood_BG(umap_BG, Sigma_map_BG, s, x);
                end
                PCheetagivenX = exp(logPxgivenCheeta + log(P_FG));
                PGrassgivenX = exp(logPxgivenGrass + log(P_BG));
                
                Img_decision(i,j) = PCheetagivenX >= PGrassgivenX;
                cnt = cnt + 1;
            end
        end

        % Compute Error Mask
        cheetah_mask = imread('cheetah_mask.bmp');
        cheetah_mask = im2bw(cheetah_mask);
        [numRows, numCols] = size(cheetah_mask);
        errors = 0;

        for i = 1:numRows
            for j = 1:numCols
                if Img_decision(i,j) ~= cheetah_mask(i,j)
                    errors = errors + 1;
                end
            end
        end
        
        % Calculate Percent Error
        PercentError = (errors / (numRows * numCols)) * 100;
        
        % Store the result in the table
        alpha_percent_errors{alpha_idx, 'Alpha'} = alpha_val;
        alpha_percent_errors{alpha_idx, 'PercentError'} = PercentError;
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
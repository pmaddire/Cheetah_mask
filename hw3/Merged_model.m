% Load the data
load("Alpha.mat");
alpha_percent_errors1 = compute_errors_for_all_alpha(alpha, true);  % Plot with specific style
alpha_percent_errors2 = compute_errors_for_all_alpha_D4(alpha, false);  % Plot with specific style

disp(alpha_percent_errors1);
disp(alpha_percent_errors2);

function alpha_percent_errors = compute_errors_for_all_alpha(alpha_values, isFirstPlot)
    % Load necessary data
    load('TrainingSamplesDCT_subsets_8.mat');
    load('Prior_2.mat');
    img = imread('cheetah.bmp');
    
    % Preallocate storage for results
    num_alphas = length(alpha_values);
    alpha_percent_errors = table('Size', [num_alphas, 2], 'VariableTypes', {'double', 'double'}, ...
                                 'VariableNames', {'Alpha', 'PercentError'});

    % Loop over each alpha value
    for alpha_idx = 1:num_alphas
        alpha_val = alpha_values(alpha_idx);
        
        % Compute Sigma_0
        Sigma_0 = alpha_val * diag(W0);
        
        % Training Data
        Train_FG = D4_FG;
        Train_BG = D4_BG;
        
        % Compute sample statistics
        FG_sample_mean = mean(Train_FG, 1);
        FG_sample_cov = cov(Train_FG, 1);
        BG_sample_mean = mean(Train_BG, 1);
        BG_sample_cov = cov(Train_BG, 1);
        
        % Posterior mean and covariance for FG
        N_FG = size(Train_FG, 1);
        x_FG = inv((Sigma_0 + (1 / N_FG) * FG_sample_cov));
        FG_posterior_mean = Sigma_0 * x_FG * FG_sample_mean' + (1 / N_FG) * FG_sample_cov * x_FG * mu0_FG';
        FG_posterior_sigma = Sigma_0 * x_FG * FG_sample_cov / N_FG + FG_sample_cov;
        
        % Posterior mean and covariance for BG
        N_BG = size(Train_BG, 1);
        x_BG = inv((Sigma_0 + (1 / N_BG) * BG_sample_cov));
        BG_posterior_mean = Sigma_0 * x_BG * BG_sample_mean' + (1 / N_BG) * BG_sample_cov * x_BG * mu0_BG';
        BG_posterior_sigma = Sigma_0 * x_BG * BG_sample_cov / N_BG + BG_sample_cov;

        % Calculate Percent Error
        % (Placeholder for error computation logic, reuse existing logic here)
        % Replace the below with the decision-making process and error calculation

        PercentError = rand(); % Placeholder, compute actual error
        alpha_percent_errors{alpha_idx, 'Alpha'} = alpha_val;
        alpha_percent_errors{alpha_idx, 'PercentError'} = PercentError;
    end
    
    % Plotting
    hold on; % Allow multiple plots on the same figure
    if isFirstPlot
        semilogx(alpha_percent_errors.Alpha, alpha_percent_errors.PercentError, '-o', 'DisplayName', 'Method 1');
    else
        semilogx(alpha_percent_errors.Alpha, alpha_percent_errors.PercentError, '-x', 'DisplayName', 'Method 2');
    end
    xlabel('Alpha');
    ylabel('Percent Error (%)');
    title('Log-Scale Plot of Percent Error vs Alpha');
    grid on;
    legend;
end

function alpha_percent_errors = compute_errors_for_all_alpha_D4(alpha_values, isFirstPlot)
    % Similar structure to compute_errors_for_all_alpha, with differences as needed
    % Add your specific logic here (reuse logic, but this is where Method 2 is implemented)

    % Placeholder for computations
    alpha_percent_errors = table('Size', [length(alpha_values), 2], 'VariableTypes', {'double', 'double'}, ...
                                 'VariableNames', {'Alpha', 'PercentError'});
    alpha_percent_errors.PercentError = rand(length(alpha_values), 1); % Replace with actual computation

    % Plotting
    hold on; % Allow multiple plots on the same figure
    if isFirstPlot
        semilogx(alpha_percent_errors.Alpha, alpha_percent_errors.PercentError, '-o', 'DisplayName', 'Method 1');
    else
        semilogx(alpha_percent_errors.Alpha, alpha_percent_errors.PercentError, '-x', 'DisplayName', 'Method 2');
    end
    xlabel('Alpha');
    ylabel('Percent Error (%)');
    title('Log-Scale Plot of Percent Error vs Alpha');
    grid on;
    legend;
end

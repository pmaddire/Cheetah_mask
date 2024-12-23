function [p_error, A] = bayes_parameter_estimate(train_cheetah, train_grass, cheetah_mu_0, cheetah_Sigma_0, grass_mu_0, grass_Sigma_0, freqs, cheetah_mask)
    N = size(train_cheetah, 1);
    cheetah_sample_mean = mean(train_cheetah, 1);
    cheetah_sample_cov = cov(train_cheetah, 1);
    cheetah_intermediate_inv = pinv(cheetah_Sigma_0 + (1 / N) * cheetah_sample_cov);
    cheetah_Bayes_mean = cheetah_Sigma_0 * cheetah_intermediate_inv * cheetah_sample_mean' + ...
                                (1 / N) * cheetah_sample_cov * cheetah_intermediate_inv * cheetah_mu_0;
    cheetah_Bayes_cov = cheetah_Sigma_0 * cheetah_intermediate_inv * cheetah_sample_cov / N + cheetah_sample_cov;
    cheetah_Bayes_cov_inv = pinv(cheetah_Bayes_cov);
    
    log_cheetah_det = sum(log(eig(cheetah_Bayes_cov)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    N = size(train_grass,1);
    grass_sample_mean = mean(train_grass, 1);
    grass_sample_cov = cov(train_grass, 1);
    grass_intermediate_inv = pinv(grass_Sigma_0 + (1 / N) * grass_sample_cov);
    grass_Bayes_mean = grass_Sigma_0 * grass_intermediate_inv * grass_sample_mean' + ...
                                (1 / N) * grass_sample_cov * grass_intermediate_inv * grass_mu_0;
    grass_Bayes_cov = grass_Sigma_0 * grass_intermediate_inv * grass_sample_cov / N + grass_sample_cov;
    grass_Bayes_cov_inv = pinv(grass_Bayes_cov);
    
    log_grass_det = sum(log(eig(grass_Bayes_cov)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    p_y_cheetah = size(train_cheetah, 1);
    p_y_grass = size(train_grass, 1);
    
    p_y_grass = p_y_grass / (p_y_cheetah + p_y_grass);
    p_y_cheetah = 1 - p_y_grass;
    
    A = zeros(size(cheetah_mask));
    
    for i = 1:size(cheetah_mask, 1)
        for j = 1:size(cheetah_mask, 2)
            dist_cheetah = (squeeze(freqs(i,j,:)) - cheetah_Bayes_mean);
            dist_grass = (squeeze(freqs(i,j,:)) - grass_Bayes_mean);
            cheetah_likelihood = -1/2 * dist_cheetah' * cheetah_Bayes_cov_inv * dist_cheetah - log_cheetah_det / 2 + log(p_y_cheetah);
            grass_likelihood = -1/2 * dist_grass' * grass_Bayes_cov_inv * dist_grass - log_grass_det / 2 + log(p_y_grass);
            if(cheetah_likelihood > grass_likelihood)
                A(i,j) = 1;
            end
        end
    end
    p_error = sum(sum(abs(A - cheetah_mask))) / (numel(cheetah_mask));
end
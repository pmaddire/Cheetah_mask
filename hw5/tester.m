
%% Load Data
load('TrainingSamplesDCT_8_new.mat'); % Load training samples
load('freqs.mat');

Data_array = freqs;

num_mix = 5;
C = 8;
dim = [1 2 4 6 8 16 24 32 40 48 56 64];

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
C = 8;
for i = 1:5
    [FG_mixprob, FG_mean, FG_sigma] = learn_mix(Train_FG, max_atemp, C, tolerance);
    [BG_mixprob, BG_mean, BG_sigma] = learn_mix(Train_BG, max_atemp, C, tolerance);
    FG_mix{i} = {FG_mean, FG_sigma, FG_mixprob};
    BG_mix{i} = {BG_mean, BG_sigma, BG_mixprob};
end
%%
p_errors = zeros(5,5,11);
for i =1:5
    i
    for j =1:5
        j
        for D=dim
            FG_predict = likelihoods(P_FG, FG_mix{j}{1}(:, 1:D), FG_mix{j}{2}(1:D,:), FG_mix{j}{3},Data_array);
            BG_predict = likelihoods(P_BG, BG_mix{i}{1}(:, 1:D), BG_mix{i}{2}(1:D,:), BG_mix{i}{3},Data_array);
            prediction = (FG_predict > BG_predict);

            error_mat = (prediction ~= cheetah_mask);
            p_errors (i, j, dim==D) = mean(error_mat,'all');
        end
    end
end
save Q5.mat p_errors

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
p_errors = zeros(length(C), length(dim));
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
save Qb.mat p_errors

%%
load('quiz5a.mat')
perrors = errors;

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
    %filestr = sprintf('TeX/images/BG Mixture %d', i);
    %saveas(gcf, filestr, 'epsc');
end

%%
%%
load quiz5b.mat

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
function [pi_mix, means, covar] = learn_mix(x, max_atemp, C, tolerance)
    C;
    %size(Dat_FG(:,1))
    [n,dim] = size(x);

    H = zeros(n,C);
    %mean = rand(n, C);  %C K components, each with n-dimensional mean vector
    means = x(randi(n, [C 1]), :);
    %size(mean)
    covar = rand(dim,C);
    %size(covar)
    pi_mix = rand(1,C);
    
    pi_mix = pi_mix/ sum(pi_mix);
    %disp("Size:")
    %size(pi_mix)

    change = 1+tolerance;
    
    count = 0
    while(change > tolerance && count< max_atemp )

    old_mean = means;
    old_covar = covar;
    old_H = H;

    H = E_step(x, means, covar, pi_mix);
    [means, covar, pi_mix] = M_step(x, H, means);

    change = sum(abs(old_mean - means), 'all') ./ sum(abs(means),'all')+ sum(abs(old_H-H),'all') ./sum(abs(H),'all') + ...
         sum(abs(old_covar-covar),'all') ./sum(abs(covar),'all');
    count = count +1

    end
end

function h = E_step(x, means, covar, pi_mix)
    c = length(pi_mix);
    n = length(x(:,1));
    h = zeros(n,c);
    
    for i=1:c
        size(x);
        size(means);
        diff = (x - means(i,:)).^2;
        
        size(covar);
        h(:,i) = -0.5 * (diff*(1 ./ covar(:,i))) - 0.5 * sum(log(covar(:,i))) + log(pi_mix(i));
    end 
    
    h = exp(h);
    h = h./ sum(h,2);
    
    
end



function [means, covar, pi_mix] = M_step(x, h, means)
    c = size(h, 2);
    n = length(x(1,:));
    
    size(mean(h,1));
    pi_mix = mean(h,1);
    covar = zeros(n, c);

    for k =1:c
        norm = sum(h(:,k)); 
        means(k,:) = sum(h(:,k) .* x)/norm;
        diff = (x - means(k,:)) .^2;
        covar(:,k) = sum(diff .* h(:,k))/norm;
    end 
    covar = covar + 1e-5; 
end


function [likelihoods] = likelihoods(prior, means, covar, pi_mix, x)
    [C, dim] = size(means);  % Number of components and feature dimension
    likelihoods = zeros(size(x, [1 2]));  % Initialize likelihood matrix
    for k = 1:size(x, 1)
        for l = 1:size(x,2)
            prob =0;
            for c = 1:C
                diff = (squeeze(x(k, l, 1:dim))' - means(c, :)) .^2;
                temp = log(pi_mix(c)) - 0.5 * (diff * (1 ./ covar(:,c))) -0.5 * sum(log(covar(:,c)));
                prob = prob + exp(temp);
            end
            likelihoods(k,1) = log(prob) + log(prior);
        end
    end 
end 











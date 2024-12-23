load('TrainingSamplesDCT_8_new.mat');

N_FG = size(TrainsampleDCT_FG,1);
N_BG = size(TrainsampleDCT_BG,1);

%% Part A
%Prior calculations: 
total = N_FG+N_BG;
P_FG = N_FG/total;
P_BG = N_BG/total;
load('data.mat')
%% Part B

%full plots
mu_cheetah = mean(TrainsampleDCT_FG);
cov_cheetah = cov(TrainsampleDCT_FG); % 64x64 matrix

% Calculate mean and covariance for the grass class
mu_grass = mean(TrainsampleDCT_BG); % 1x64 vector
cov_grass = cov(TrainsampleDCT_BG); % 64x64 matrix

% Prepare a grid of values for plotting
x_values = linspace(-0.3, 0.3, 100);  % Adjust range as needed
num_features = 64;

% Preallocate matrices for densities
densities_cheetah = zeros(num_features, length(x_values));
densities_grass = zeros(num_features, length(x_values));

for k = 1:num_features
    mu_c = mu_cheetah(k);
    sigma_c = sqrt(cov_cheetah(k, k));
    densities_cheetah(k, :) = (1/(sqrt(2*pi)*sigma_c)) * exp(-0.5 * ((x_values - mu_c) ./ sigma_c).^2);

    mu_g = mu_grass(k);
    sigma_g = sqrt(cov_grass(k, k));
    densities_grass(k, :) = (1/(sqrt(2*pi)*sigma_g)) * exp(-0.5 * ((x_values - mu_g) ./ sigma_g).^2);
end

% Create a figure for plotting
figure('Position', [50, 50, 1500, 1000]); % Increase figure size
m = 48;
% Create a grid of subplots (4x4) for larger plots
for t = 1:num_features
    subplot(4, 4, t); % Creates a grid of subplots (4x4)
    k = t+m;
    plot(x_values, densities_cheetah(k, :), 'b-', 'DisplayName', 'Cheetah');
    hold on;
    plot(x_values, densities_grass(k, :), 'r--', 'DisplayName', 'Grass');

    % Add vertical lines for the means
    y_lim = ylim; % Get current y-limits for the plot
    plot([mu_cheetah(k), mu_cheetah(k)], y_lim, 'b:', 'LineWidth', 1.5, 'DisplayName', 'Mean (Cheetah)');
    plot([mu_grass(k), mu_grass(k)], y_lim, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Mean (Grass)');

    % Annotate the mean values on the plots
    text(mu_cheetah(k), max(densities_cheetah(k, :)), sprintf('%.2f', mu_cheetah(k)), 'Color', 'blue', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    text(mu_grass(k), max(densities_grass(k, :)), sprintf('%.2f', mu_grass(k)), 'Color', 'red', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

    title(['Feature ', num2str(k)]);
    xlabel('DCT Coefficient Value');
    ylabel('Density');
    legend('Location', 'best');
    hold off;
end

%% Part B

%specific plots
mu_cheetah = mean(TrainsampleDCT_FG);
cov_cheetah = cov(TrainsampleDCT_FG); % 64x64 matrix

% Calculate mean and covariance for the grass class
mu_grass = mean(TrainsampleDCT_BG); % 1x64 vector
cov_grass = cov(TrainsampleDCT_BG); % 64x64 matrix

% Prepare a grid of values for plotting
x_values = linspace(-0.1, 0.1, 100);  % Adjust range as needed
num_features = 64;

% Preallocate matrices for densities
densities_cheetah = zeros(num_features, length(x_values));
densities_grass = zeros(num_features, length(x_values));

for k = 1:num_features
    mu_c = mu_cheetah(k);
    sigma_c = sqrt(cov_cheetah(k, k));
    densities_cheetah(k, :) = (1/(sqrt(2*pi)*sigma_c)) * exp(-0.5 * ((x_values - mu_c) ./ sigma_c).^2);

    mu_g = mu_grass(k);
    sigma_g = sqrt(cov_grass(k, k));
    densities_grass(k, :) = (1/(sqrt(2*pi)*sigma_g)) * exp(-0.5 * ((x_values - mu_g) ./ sigma_g).^2);
end

% Create a figure for plotting
figure('Position', [50, 50, 1500, 1000]); % Increase figure size
m = 32;
ls = [36,37,57,59,60,62,63,64];
% Create a grid of subplots (4x4) for larger plots
for t = 1:length(ls)
    subplot(2, 4, t); % Creates a grid of subplots (4x4)
    k = ls(t);
    plot(x_values, densities_cheetah(k, :), 'b-', 'DisplayName', 'Cheetah');
    hold on;
    plot(x_values, densities_grass(k, :), 'r--', 'DisplayName', 'Grass');

    % Add vertical lines for the means
    y_lim = ylim; % Get current y-limits for the plot
    plot([mu_cheetah(k), mu_cheetah(k)], y_lim, 'b:', 'LineWidth', 1.5, 'DisplayName', 'Mean (Cheetah)');
    plot([mu_grass(k), mu_grass(k)], y_lim, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Mean (Grass)');

    % Annotate the mean values on the plots
    text(mu_cheetah(k), max(densities_cheetah(k, :)), sprintf('%.2f', mu_cheetah(k)), 'Color', 'blue', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    text(mu_grass(k), max(densities_grass(k, :)), sprintf('%.2f', mu_grass(k)), 'Color', 'red', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

    title(['Feature ', num2str(k)]);
    xlabel('DCT Coefficient Value');
    ylabel('Density');
    legend('Location', 'best');
    hold off;
end

%% Part C
%dct part
img = imread('cheetah.bmp');
img = im2double(img);
zigZag = load('Zig-Zag Pattern.txt'); 
[numRows, numCols] = size(img);
%define our pixel as the (4,4) pixel+
boxSize = [8,8];
%zero_rowsT = zeros(3,numCols);
zero_rowsT = img(1,:);
new_arrayT = repmat(zero_rowsT, 3, 1); % Repeat the row 3 times
%zero_rowsB = zeros(4,numCols);
zero_rowsB = img(end,:);
new_arrayB = repmat(zero_rowsB, 4, 1);
rowpack_img = [img;new_arrayB];
rowpack_img = [new_arrayT; rowpack_img];

[numRows, numCols] = size(rowpack_img);
%zero_colL = zeros(numRows, 3);
zero_colL = rowpack_img(:, 1);
new_arrayL = repmat(zero_colL, 1, 3);
%zero_colR = zeros(numRows, 4);
zero_colR = rowpack_img(:, end);
new_arrayR = repmat(zero_colR, 1, 4);

imgpack = transpose([transpose(new_arrayL);transpose(rowpack_img)]);
imgpack = transpose([transpose(imgpack);transpose(new_arrayR)]);
blockSize =[8,8];
[numRows, numCols] = size(img);
dctBlocks_full = zeros(numRows*8,numCols*8);
% A1 = dctBlocks(1+rdx:blockSize(1)+rdx,1+cdx:blockSize(2)+cdx);
% Data_array = data_fill(blockSize, zigZag, A1, block_num, Data_array);
Data_array = zeros(numRows*numCols, blockSize(1)*blockSize(2));
check_array = zeros(numRows,numCols);
[numRows, numCols] = size(imgpack);
rdx =0;
cdx =0;
cnt = 1;
for i = 1:numRows-7
    rdx = i
    for j = 1:numCols-7
        cdx = j;
        %A1 = dctBlocks(1+rdx:blockSize(1)+rdx,1+cdx:blockSize(2)+cdx);
        A1 = blockproc(imgpack(rdx:blockSize(1)-1+rdx,cdx:blockSize(2)+cdx-1), blockSize, @(block) dct2(block.data));
        Data_array = data_fill(blockSize, zigZag,A1,cnt, Data_array);
        check_array(i,j) = 1;
        cnt = cnt+1;
    end
end
%% Classification part for 64 Gaussians 
img = imread('cheetah.bmp');
[numRows, numCols] = size(img);
Img_decision = zeros(numRows, numCols);
cnt = 1; 
for i = 1:numRows 
    for j = 1:numCols
        logPxgivenCheeta = 0;
        logPxgivenGrass = 0;
        for s = 1:64
            x = Data_array(cnt, s);
            logPxgivenCheeta = logPxgivenCheeta + likliehood_FG(mu_cheetah, cov_cheetah, s, x);
            logPxgivenGrass = logPxgivenGrass + likliehood_BG(mu_grass, cov_grass, s, x);
        end
        PCheetagivenX = exp(logPxgivenCheeta + log(P_FG));
        PGrassgivenX = exp(logPxgivenGrass + log(P_BG));
        
        Img_decision(i,j) = PCheetagivenX >= PGrassgivenX;
        cnt = cnt + 1;
    end
end

%% classification for 8 Gaussians 
[numRows, numCols] = size(img);
Img_decision = zeros(numRows, numCols);
ids = [1, 7, 8, 9, 12, 14, 17, 25];
%ids = [1, 25, 3, 4, 7, 8, 14, 17];
cnt = 1; 
for i = 1:numRows 
    for j = 1:numCols
        logPxgivenCheeta = 0;
        logPxgivenGrass = 0;
        for s = 1:length(ids)
            idx = ids(s);
            x = Data_array(cnt, idx);
            logPxgivenCheeta = logPxgivenCheeta + likliehood_FG(mu_cheetah, cov_cheetah, idx, x);
            logPxgivenGrass = logPxgivenGrass + likliehood_BG(mu_grass, cov_grass, idx, x);
        end
        PCheetagivenX = exp(logPxgivenCheeta + log(P_FG));
        PGrassgivenX = exp(logPxgivenGrass + log(P_BG));
        
        Img_decision(i,j) = PCheetagivenX >= PGrassgivenX;
        cnt = cnt + 1;
    end
end

A =Img_decision;

%% create mask 

A =Img_decision;
figure;
imagesc(A);
colormap(gray(225));  
           
title('Decision Mask for Cheetah');

%% ERROR

cheetah_mask = imread('cheetah_mask.bmp');  
cheetah_mask = im2bw(cheetah_mask);         

[numRows, numCols] = size(cheetah_mask);
errors = 0;
for i = 1:numRows
    for j = 1:numCols
        if A(i,j) ~= cheetah_mask(i,j)
            errors = errors +1;
        end 
    end
end

Percenterror = errors/(numRows*numCols)
figure;
imagesc(cheetah_mask);
colormap(gray(225)); 

%% Part C function
function Data_array = data_fill(blockSize, zigZag,A1,block_num, Data_array)
    
    for i = 1:blockSize(2)
        for j = 1:blockSize(1)
            idx = zigZag(i, j) + 1; 
            Data_array(block_num, idx) = A1(i, j);  
        end
    end
end

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


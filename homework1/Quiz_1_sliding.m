load('TrainingSamplesDCT_8.mat');

N_FG = size(TrainsampleDCT_FG,1);
N_BG = size(TrainsampleDCT_BG,1);

%% Part A
%Prior calculations: 
total = N_FG+N_BG;
P_FG = N_FG/total;
P_BG = N_BG/total;

%% Part B
FG_histogramdat = zeros(1,64);
BG_histogramdat = zeros(1,64);

for i = 1:N_FG
    x= secondbiggest(TrainsampleDCT_FG(i,:));
    FG_histogramdat(x) = FG_histogramdat(x)+1; 

end

for i = 1:N_BG
    x= secondbiggest(TrainsampleDCT_BG(i,:));
    BG_histogramdat(x) = BG_histogramdat(x)+1; 

end

figure; 
h = histogram('BinEdges', 1:65, 'BinCounts', FG_histogramdat,Normalization="percentage");
title("P(x|Cheetah)")
ylabel("Percent chance")
xlabel("Second largest Index")
ytickformat("percentage")
figure;

h = histogram('BinEdges', 1:65, 'BinCounts', BG_histogramdat, Normalization="percentage");
title("P(x|Grass)")
ylabel("Percent chance")
xlabel("Second largest Index")
ytickformat("percentage")

%%

img = imread('cheetah.bmp');
img = im2double(img);
zigZag = load('Zig-Zag Pattern.txt'); 
[numRows, numCols] = size(img);
%define our pixel as the (4,4) pixel+
boxSize = [8,8];
zero_rowsT = zeros(3,numCols);
zero_rowsB = zeros(4,numCols);
rowpack_img = [img;zero_rowsB];
rowpack_img = [zero_rowsT; rowpack_img];
[numRows, numCols] = size(rowpack_img);
zero_colL = zeros(numRows, 3);
zero_colR = zeros(numRows, 4);
imgpack = transpose([transpose(zero_colL);transpose(rowpack_img)]);
imgpack = transpose([transpose(imgpack);transpose(zero_colR)]);
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
%dctBlocks = blockproc(imgpack(1:8,1:8), blockSize, @(block) dct2(block.data));
%%

[numRows, numCols] = size(img);
Img_decision = zeros(numRows, numCols);
cnt = 1; 
for i = 1:numRows 
    for j = 1:numCols
        idx = secondbiggest(Data_array(cnt,:));
        PxgivenCheeta = likliehood_FG(FG_histogramdat, N_FG, idx);
        PxgivenGrass = likliehood_BG(BG_histogramdat, N_BG, idx);

        PCheetagivenX = PxgivenCheeta * P_FG;
        PGrassgivenX = PxgivenGrass * P_BG;
        if PCheetagivenX >= PGrassgivenX
            Img_decision(i,j) = 1;
        
        else 
            Img_decision(i,j) = 0;
        end
        %disp(['Block: ', num2str(cnt), ' | PxgivenCheetah: ', num2str(PxgivenCheeta), ' | PxgivenGrass: ', num2str(PxgivenGrass)]);

        cnt = cnt +1;

    end
end
%%

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


%% function
function f = secondbiggest(n)
    x = sort(n, 'descend');
    t = x(2);
    f = find(n==t);
end


%% Part C function
function Data_array = data_fill(blockSize, zigZag,A1,block_num, Data_array)
    
    for i = 1:blockSize(2)
        for j = 1:blockSize(1)
            idx = zigZag(i, j) + 1; 
            Data_array(block_num, idx) = A1(i, j);  
        end
    end
end

function PxgivenCheeta = likliehood_FG(FG_histogramdat, N_FG, idx) 
    x = FG_histogramdat(idx);
    PxgivenCheeta = x/N_FG;

end

function PxgivenGrass = likliehood_BG(BG_histogramdat, B_FG, idx) 
    x = BG_histogramdat(idx);
    PxgivenGrass = x/B_FG;
    
end


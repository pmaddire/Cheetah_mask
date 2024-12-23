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



%% Part C

img = imread('cheetah.bmp');
img = im2double(img);
zigZag = load('Zig-Zag Pattern.txt'); 


blockSize = [8, 8];


[numRows, numCols] = size(img);
numBlocksVertical = numRows / blockSize(1);
numBlocksHorizontal = numCols / blockSize(2);

x = mod(numRows,blockSize(1)); 
Zero_rows = zeros(blockSize(1)-x,numCols);
rowpack_img = [img;Zero_rows];


x = mod(numCols, blockSize(2));
[numRows, numCols] = size(rowpack_img);

Zero_Columns = zeros(numRows, blockSize(2)-x);

packed_img = transpose([transpose(rowpack_img);transpose(Zero_Columns)]);
[numRows, numCols] = size(packed_img);
%mod(numRows,8)
%mod(numCols, 8)
dctBlocks = blockproc(packed_img, blockSize, @(block) dct2(block.data));
%%

num_blocks = (numRows/blockSize(1))*(numCols/blockSize(2));

%for first block
%data_fill(blockSize,zigZag);
%Data_array = data_fill(blockSize, zigZag,A1,block_num, Data_array)
%assign block numbers
Data_array = zeros(num_blocks, blockSize(1)*blockSize(2));
counter = 1;
rcount = 1;
rdx = 0;
block_num = 1;
blocksperRow = numCols/blockSize(2);
for i = 1:num_blocks
    cdx = ((counter-1)*blockSize(1));
    A1 = dctBlocks(1+rdx:blockSize(1)+rdx,1+cdx:blockSize(2)+cdx);
    Data_array = data_fill(blockSize, zigZag, A1, block_num, Data_array);
    if(mod(counter,blocksperRow)==0)
        cdx = 0;
        counter = 0;  
        rcount = rcount+1;
        rdx = rdx+blockSize(1);
    end
    
    block_num = block_num+1;
    counter = counter+1;

end

blocksperCol = numRows/blockSize(1);
Img_decision = zeros(blocksperCol, blocksperRow);
cnt = 1; 
for i = 1:blocksperCol 
    for j = 1:blocksperRow
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

%function PxgivenCheeta = likliehood_FG(FG_histogramdat, N_FG, idx) 
A =Img_decision;




scaled_A = kron(A, ones(8, 8));  
corrected_A = scaled_A(1:end-1,1:end-2);% have to change this if I want to work with other block dim

figure;
imagesc(corrected_A);
colormap(gray(225));  
axis equal;           
title('Scaled Decision Mask for Cheetah');

%% ERROR

A = corrected_A;


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

%% Part B function


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








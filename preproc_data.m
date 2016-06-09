load('MNIST.mat');
% test_img = gpuArray(single(reshape(test_data(:,:),size(test_data,1),inputSize,inputSize)));
test_img = single(reshape(test_data(:,:),size(test_data,1),inputSize,inputSize));
test_img = permute(test_img,[1 3 2]);
% test_label = gpuArray(test_label);
clear test_data

% train_img = gpuArray(single(reshape(train_data(:,:),60000,inputSize,inputSize)));
train_img = single(reshape(train_data(:,:),60000,inputSize,inputSize));
train_img = permute(train_img,[1 3 2]);
% train_label = gpuArray(train_label);
clear train_data

img_mean = mean(train_img,1);
train_img = train_img - repmat(img_mean, [size(train_img,1) 1 1]);
test_img = test_img - repmat(img_mean, [size(test_img,1) 1 1]);
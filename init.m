clear all;  gdv=gpuDevice(1); reset(gdv); wait(gdv);

inputSize = 28;

load('MNIST.mat');
test_img = gpuArray(single(reshape(test_data(:,:),size(test_data,1),inputSize,inputSize)));
test_img = permute(test_img,[1 3 2]);
clear test_data

train_img = gpuArray(single(reshape(train_data(:,:),60000,inputSize,inputSize)));
train_img = permute(train_img,[1 3 2]);
clear train_data

img_mean = mean(train_img,1);
train_img = train_img - repmat(img_mean, [size(train_img,1) 1 1]);
test_img = test_img - repmat(img_mean, [size(test_img,1) 1 1]);

opt.solver.lr = 0.2;
opt.solver.batchsize = 100;
opt.solver.epoch = 3;
opt.solver.weight_decay = 0.0001;
opt.solver.momentum = 0.95;
opt.solver.pooling = 'MAX';
opt.solver.inputsize = inputSize;
opt.solver.verbose = false;

opt.layer(1).type='CONV';
opt.layer(1).num_output=20;
opt.layer(1).kernel=5;
opt.layer(1).stride=1;
opt.layer(1).padding=0;

opt.layer(2).type='POOL';
opt.layer(2).kernel=2;
opt.layer(2).stride=2;
opt.layer(2).padding=0;

opt.layer(3).type='CONV';
opt.layer(3).num_output=50;
opt.layer(3).kernel=5;
opt.layer(3).stride=1;
opt.layer(3).padding=0;

opt.layer(4).type='POOL';
opt.layer(4).kernel=2;
opt.layer(4).stride=2;
opt.layer(4).padding=0;

opt.layer(5).type='FC';
opt.layer(5).num_output=500;

opt.layer(6).type='RELU';

opt.layer(7).type='FC';
opt.layer(7).num_output=10;

[model, opt] = makeModel(opt);
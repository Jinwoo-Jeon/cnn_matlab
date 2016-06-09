clear model opt;



opt.solver.baselr = 0.01;
opt.solver.lr = opt.solver.baselr;
opt.solver.gamma = 0.001;
opt.solver.power = 0.75;
opt.solver.batchsize = 60;
opt.solver.epoch = 1;
opt.solver.weight_decay = 0.0005;
opt.solver.momentum = 0.9;
opt.solver.pooling = 'MAX';
opt.solver.inputsize = inputSize;
opt.solver.verbose = false;
opt.solver.testPeriod = 10;
opt.solver.savePeriod = 250;
opt.solver.startepoch = 1;
opt.solver.startiter = 1;
opt.solver.costArr = [];
opt.solver.errArr = [];












ind=1;
opt.layer(ind).type='CONV';
opt.layer(ind).num_output=20;
opt.layer(ind).kernel=5;
opt.layer(ind).stride=1;
opt.layer(ind).padding=0;

ind=ind+1;
opt.layer(ind).type='RELU';

ind=ind+1;
opt.layer(ind).type='POOL';
opt.layer(ind).kernel=2;
opt.layer(ind).stride=2;
opt.layer(ind).padding=0;

ind=ind+1;
opt.layer(ind).type='CONV';
opt.layer(ind).num_output=50;
opt.layer(ind).kernel=5;
opt.layer(ind).stride=1;
opt.layer(ind).padding=0;

ind=ind+1;
opt.layer(ind).type='PRELU';

ind=ind+1;
opt.layer(ind).type='POOL';
opt.layer(ind).kernel=2;
opt.layer(ind).stride=2;
opt.layer(ind).padding=0;

ind=ind+1;
opt.layer(ind).type='FC';
opt.layer(ind).num_output=500;

ind=ind+1;
opt.layer(ind).type='PRELU';

ind=ind+1;
opt.layer(ind).type='FC';
opt.layer(ind).num_output=10;

ind=ind+1;
opt.layer(ind).type='SOFTMAX';







[model, opt] = makeModel(opt);
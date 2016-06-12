% layer output: 
% 	batch x width x height x chan 
% 
% weight output: 
% 	batch x width x height x in_chan x out_chan
inputSize = 28;
% gdv=gpuDevice(1); reset(gdv); wait(gdv);

disp('loading data...')
% preproc_data;

disp('init model...')
% init;




opt.solver.verbose = false;
% opt.solver.verbose = true;

train(model, opt, train_img, train_label, test_img, test_label);
% test(model, opt, test_img, test_label);
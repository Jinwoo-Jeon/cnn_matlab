function res = train(model_cnn, option, data, label)

cur_epoch = 0;
train_num = size(data,1);
batch_num = option.solver.batchsize;
rand_train_idx = randperm(train_num);
while cur_epoch<option.solver.epoch
    for i=1:floor(train_num/batch_num)
        cnn_res = forward(model_cnn, option, data(rand_train_idx((i-1)*batch_num+1:i*batch_num),:,:));
        error = label(rand_train_idx((i-1)*batch_num+1:i*batch_num),:)-squeeze(cnn_res);
        
        
        
    end
    


    cur_epoch = cur_epoch+1;
end

end